"""LBM 单张图片重打光推理与测速脚本.

与官方 gradio_demo 的流程一致：抠出主体 -> 居中裁剪贴到目标背景 -> LBM 单步推理 ->
将输出主体贴回背景。mask 需提前准备好（BiRefNet/rembg 均可）。
"""

import argparse
import os
import time

import torch
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

from lbm.inference import get_model
from utils import resize_and_center_crop

PATH = os.path.dirname(os.path.abspath(__file__))

ASPECT_RATIOS = {
    str(512 / 2048): (512, 2048),
    str(1024 / 1024): (1024, 1024),
    str(2048 / 512): (2048, 512),
    str(896 / 1152): (896, 1152),
    str(1152 / 896): (1152, 896),
    str(512 / 1920): (512, 1920),
    str(640 / 1536): (640, 1536),
    str(768 / 1280): (768, 1280),
    str(1280 / 768): (1280, 768),
    str(1536 / 640): (1536, 640),
    str(1920 / 512): (1920, 512),
}


def load_model(model_dir, device="cuda"):
    t0 = time.time()
    model = get_model(model_dir, torch_dtype=torch.bfloat16, device=device)
    return model, time.time() - t0


@torch.no_grad()
def relight(model, fg_path, mask_path, bg_path, num_steps=1):
    fg_image = Image.open(fg_path).convert("RGB")
    fg_mask = Image.open(mask_path).convert("L")
    bg_image = Image.open(bg_path).convert("RGB")

    ori_w, ori_h = fg_image.size
    ar = ori_h / ori_w
    closest = min(ASPECT_RATIOS, key=lambda x: abs(float(x) - ar))
    w, h = ASPECT_RATIOS[closest]

    fg_image = resize_and_center_crop(fg_image, w, h)
    fg_mask = resize_and_center_crop(fg_mask, w, h)
    bg_image = resize_and_center_crop(bg_image, w, h)

    img_pasted = Image.composite(fg_image, bg_image, fg_mask)

    source_tensor = (ToTensor()(img_pasted).unsqueeze(0) * 2 - 1).cuda().to(torch.bfloat16)

    t0 = time.time()
    z_source = model.vae.encode(source_tensor)
    output = model.sample(
        z=z_source,
        num_steps=num_steps,
        conditioner_inputs={"source_image": source_tensor},
        max_samples=1,
    ).clamp(-1, 1)
    latency = time.time() - t0

    output = (output[0].float().cpu() + 1) / 2
    output = ToPILImage()(output)
    output = Image.composite(output, bg_image, fg_mask)

    return img_pasted, output, latency, (w, h)


def side_by_side(a, b):
    canvas = Image.new("RGB", (a.width + b.width, max(a.height, b.height)), (255, 255, 255))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width, 0))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=os.path.join(PATH, "ckpts", "relighting"))
    parser.add_argument("--source_image", required=True, help="主体图（前景）")
    parser.add_argument("--mask_image", required=True, help="主体 mask，白=主体")
    parser.add_argument("--background_image", required=True, help="目标背景图")
    parser.add_argument("--output_path", default="outputs")
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3, help="正式计时次数（不含预热）")
    args = parser.parse_args()

    model, load_time = load_model(args.model_dir)
    print(f"模型加载耗时: {load_time:.2f}s")
    print(f"权重显存: {torch.cuda.memory_allocated()/2**30:.2f} GiB")

    # 预热
    relight(model, args.source_image, args.mask_image, args.background_image,
            args.num_inference_steps)

    latencies = []
    result = None
    for _ in range(args.runs):
        result = relight(model, args.source_image, args.mask_image,
                         args.background_image, args.num_inference_steps)
        latencies.append(result[2])

    img_pasted, output, _, size = result
    print(f"推理分辨率: {size[0]}x{size[1]}, 步数: {args.num_inference_steps}")
    print(f"单张推理耗时: " + ", ".join(f"{t*1000:.0f}ms" for t in latencies))
    print(f"平均 {sum(latencies)/len(latencies)*1000:.0f}ms | 峰值显存 {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")

    os.makedirs(args.output_path, exist_ok=True)
    img_pasted.save(os.path.join(args.output_path, "source_composite.png"))
    output.save(os.path.join(args.output_path, "output.png"))
    side_by_side(img_pasted, output).save(os.path.join(args.output_path, "comparison.png"))
    print(f"结果已保存到: {os.path.abspath(args.output_path)}")


if __name__ == "__main__":
    main()
