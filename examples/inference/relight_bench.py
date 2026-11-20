"""LBM 精简流水线：RGB + MASK + 自定义背景 -> 打光 -> 换背景，并测速。

数据源为 DHGA-IR-packedN 的 dhgapack 容器（每文件 240 帧 JPEG）。
流程：
  1. 读 RGB 帧与 MASK，按 mask 合成到"打光背景 A"
  2. LBM 单步重打光（默认按输入分辨率 256 推理，可指定 512/1024）
  3. 用同一 mask 抠出打光后的前景，贴到"最终背景 B"
  4. 输出分辨率与输入帧一致
"""

import argparse
import functools
import io
import os
import struct
import time

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import ToPILImage, ToTensor

from lbm.inference import get_model

PATH = os.path.dirname(os.path.abspath(__file__))


@functools.lru_cache(maxsize=4)
def read_frames(pack_path):
    """解析 DHGAPK01 容器，按 JPEG 边界切出所有帧（按文件缓存）。"""
    data = open(pack_path, "rb").read()
    if data[:8] != b"DHGAPK01":
        raise ValueError(f"不是 dhgapack 文件: {pack_path}")
    pos = struct.unpack("<I", data[0x18:0x1C])[0]
    frames = []
    while pos < len(data):
        s = data.find(b"\xff\xd8\xff", pos)
        if s < 0:
            break
        e = data.find(b"\xff\xd9", s)
        if e < 0:
            break
        frames.append(data[s : e + 2])
        pos = e + 2
    return frames


def load_pair(rgb_pack, mask_pack, index):
    rgb = Image.open(io.BytesIO(read_frames(rgb_pack)[index])).convert("RGB")
    mask_raw = Image.open(io.BytesIO(read_frames(mask_pack)[index])).convert("L")
    mask = mask_raw.point(lambda v: 255 if v > 127 else 0)
    return rgb, mask


def bg_to_square(bg, size):
    return ImageOps.fit(bg, (size, size), Image.LANCZOS)


@torch.no_grad()
def relight(model, fg, mask, bg_light, bg_final, resolution, steps):
    w, h = fg.size

    fg_r = fg.resize((resolution, resolution), Image.LANCZOS)
    mask_r = mask.resize((resolution, resolution), Image.LANCZOS).point(
        lambda v: 255 if v > 127 else 0
    )
    bg_light_r = bg_to_square(bg_light, resolution)
    bg_final_r = bg_to_square(bg_final, resolution)

    composite = Image.composite(fg_r, bg_light_r, mask_r)
    x = (ToTensor()(composite).unsqueeze(0) * 2 - 1).cuda().to(torch.bfloat16)

    torch.cuda.synchronize()
    t0 = time.time()
    z = model.vae.encode(x)
    out = model.sample(
        z=z, num_steps=steps, conditioner_inputs={"source_image": x}, max_samples=1
    ).clamp(-1, 1)
    torch.cuda.synchronize()
    latency = time.time() - t0

    out = ToPILImage()((out[0].float().cpu() + 1) / 2)
    final = Image.composite(out, bg_final_r, mask_r)

    # 输出与输入分辨率一致
    if (w, h) != (resolution, resolution):
        composite = composite.resize((w, h), Image.LANCZOS)
        final = final.resize((w, h), Image.LANCZOS)

    return composite, final, latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=os.path.join(PATH, "ckpts", "relighting"))
    parser.add_argument("--rgb_pack", required=True)
    parser.add_argument("--mask_pack", required=True)
    parser.add_argument("--light_bg", default="/home/wxl3/shh/input/1.jpg", help="打光条件背景 A")
    parser.add_argument("--final_bg", default="/home/wxl3/shh/input/2.jpg", help="最终贴回去的背景 B")
    parser.add_argument("--resolution", type=int, default=256, help="LBM 推理分辨率")
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--runs", type=int, default=2, help="每帧计时次数")
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    bg_light = Image.open(args.light_bg).convert("RGB")
    bg_final = Image.open(args.final_bg).convert("RGB")

    t0 = time.time()
    model = get_model(args.model_dir, torch_dtype=torch.bfloat16, device="cuda")
    print(f"模型加载: {time.time()-t0:.1f}s | 权重显存 {torch.cuda.memory_allocated()/2**30:.2f} GiB")

    os.makedirs(args.output_path, exist_ok=True)

    # 预热
    fg, mask = load_pair(args.rgb_pack, args.mask_pack, args.start)
    relight(model, fg, mask, bg_light, bg_final, args.resolution, args.num_inference_steps)

    latencies = []
    e2e = []
    composites = []
    finals = []
    for i in range(args.start, args.start + args.count):
        t_frame = time.time()
        fg, mask = load_pair(args.rgb_pack, args.mask_pack, i)
        for _ in range(args.runs):
            comp, final, lat = relight(
                model, fg, mask, bg_light, bg_final,
                args.resolution, args.num_inference_steps,
            )
            latencies.append(lat)
        e2e.append((time.time() - t_frame) / args.runs)

        comp.save(os.path.join(args.output_path, f"frame{i:03d}_input.png"))
        final.save(os.path.join(args.output_path, f"frame{i:03d}_output.png"))
        composites.append(comp)
        finals.append(final)

    # 预览大图：上排为输入合成图，下排为最终输出
    if composites:
        w, h = composites[0].size
        grid = Image.new("RGB", (w * len(composites), h * 2))
        for j, (c, f) in enumerate(zip(composites, finals)):
            grid.paste(c, (j * w, 0))
            grid.paste(f, (j * w, h))
        grid.save(os.path.join(args.output_path, "grid.png"))

    n = len(latencies)
    lat_ms = np.array(latencies) * 1000
    print(f"推理分辨率 {args.resolution}x{args.resolution} | 步数 {args.num_inference_steps} | 帧数 {n}")
    print(f"模型推理: 平均 {lat_ms.mean():.0f}ms | 最小 {lat_ms.min():.0f}ms | 最大 {lat_ms.max():.0f}ms | {1000/lat_ms.mean():.1f} FPS")
    print(f"端到端(含读帧/合成/保存): 平均 {np.mean(e2e)*1000:.0f}ms | {1/np.mean(e2e):.1f} FPS")
    print(f"峰值显存 {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    print(f"输出目录: {os.path.abspath(args.output_path)}")


if __name__ == "__main__":
    main()
