#!/usr/bin/env bash
# LBM 精简流水线：RGB + MASK 合成到背景 A -> LBM 重打光 -> 抠前景贴到背景 B -> 测速
# 用法：
#   直接改下面的参数，然后  bash /home/wxl3/shh/LBM/bash.sh
#   也可以临时覆盖，例如： RESOLUTION=512 COUNT=16 bash /home/wxl3/shh/LBM/bash.sh
# 注意：必须在宿主机终端前台运行（沙箱内会出现 CUDA Error 304 假报错），且需保证 GPU 空闲。

set -euo pipefail

# ===================== 运行环境 =====================
# lbm 专用的 Python 解释器（conda 环境 lbm，Python 3.10 + torch 2.7.0+cu126）
PY=/home/wxl3/miniconda3/envs/lbm/bin/python
# HuggingFace 镜像，模型加载时若本地缓存缺失会从这里下载 SDXL VAE/scheduler
export HF_ENDPOINT=https://hf-mirror.com

# ===================== 模型 =====================
# LBM relighting 权重目录（含 config.yaml 与 model.safetensors）
MODEL_DIR=/home/wxl3/shh/LBM/examples/inference/ckpts/relighting

# ===================== 输入数据 =====================
# RGB 的 dhgapack 容器（每文件 240 帧 JPEG，帧号与 MASK 一一对应）
RGB_PACK=/newdisk/data/DHGA-IR-packedN/RGB/1_1_1_0_0_0.dhgapack
# MASK 的 dhgapack 容器（白色=手部前景）
MASK_PACK=/newdisk/data/DHGA-IR-packedN/MASK/1_1_1_0_0_0.dhgapack

# ===================== 背景 =====================
# 打光条件背景 A：前景先贴到这张图上送进 LBM，由它决定打光方向/色调
LIGHT_BG=/home/wxl3/shh/input/1.jpg
# 最终背景 B：打光完成后，用 MASK 抠出前景贴到这张图上输出
FINAL_BG=/home/wxl3/shh/input/2.jpg

# ===================== 推理参数 =====================
# LBM 推理分辨率（正方形）：256=输入原生尺寸最快；512/1024 质量更高但更慢
RESOLUTION=${RESOLUTION:-256}
# 采样步数：1 步是 LBM 官方推荐（单步生成），2~4 步略慢、提升有限
STEPS=${STEPS:-1}

# ===================== 测速范围 =====================
# 起始帧号（0~239）
START=${START:-0}
# 参与测速与保存的帧数
COUNT=${COUNT:-8}
# 每帧正式计时次数（不含预热），最终取平均
RUNS=${RUNS:-2}

# ===================== 输出 =====================
# 输出目录：逐帧 input/output、上下排对比图 grid.png 都在这里（默认按分辨率分目录）
OUTPUT=${OUTPUT:-/home/wxl3/shh/LBM/outputs/bench${RESOLUTION}}

# ===================== 执行 =====================
SCRIPT=/home/wxl3/shh/LBM/examples/inference/relight_bench.py

cd /home/wxl3/shh/LBM/examples/inference || exit 1

echo "=============================================="
echo " RGB_PACK   : $RGB_PACK"
echo " MASK_PACK  : $MASK_PACK"
echo " LIGHT_BG   : $LIGHT_BG"
echo " FINAL_BG   : $FINAL_BG"
echo " RESOLUTION : $RESOLUTION x $RESOLUTION"
echo " STEPS      : $STEPS"
echo " FRAMES     : start=$START count=$COUNT runs=$RUNS"
echo " OUTPUT     : $OUTPUT"
echo "=============================================="

"$PY" "$SCRIPT" \
  --model_dir "$MODEL_DIR" \
  --rgb_pack "$RGB_PACK" \
  --mask_pack "$MASK_PACK" \
  --light_bg "$LIGHT_BG" \
  --final_bg "$FINAL_BG" \
  --resolution "$RESOLUTION" \
  --num_inference_steps "$STEPS" \
  --start "$START" \
  --count "$COUNT" \
  --runs "$RUNS" \
  --output_path "$OUTPUT"
