#!/bin/bash
# Kronos 模型下载脚本
# 使用方法: bash scripts/download_kronos.sh

set -e

echo "=========================================="
echo "Kronos Model Download Script"
echo "=========================================="

# 激活 conda 环境
echo "Activating conda environment: freqtrade"
echo "=========================================="
# 初始化 conda（适配不同系统）
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "Warning: Could not find conda.sh, trying direct activation..."
    eval "$(conda shell.bash hook)"
fi

conda activate freqtrade
echo "Conda environment 'freqtrade' activated"
echo ""

# 创建目录
KRONOS_DIR="user_data/models/kronos"
KRONOS_LIB_DIR="user_data/freqaimodels/kronos_lib"

mkdir -p "$KRONOS_DIR"
mkdir -p "$KRONOS_LIB_DIR"

# 检查 huggingface-hub 是否安装
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface-hub..."
    pip install huggingface-hub
fi

# 检查 HuggingFace 登录状态
echo ""
echo "Checking HuggingFace authentication..."
echo "=========================================="
if ! huggingface-cli whoami &>/dev/null; then
    echo ""
    echo "⚠️  You are not logged in to HuggingFace!"
    echo "The Kronos model requires authentication to download."
    echo ""
    echo "Please run the following command first:"
    echo "  huggingface-cli login"
    echo ""
    echo "Or set your HuggingFace token:"
    echo "  export HF_TOKEN=your_token_here"
    echo ""
    exit 1
fi
echo "✓ HuggingFace authentication OK"

# 下载 Kronos-small 模型
echo ""
echo "Downloading Kronos-small model..."
echo "=========================================="
hf download NeoQuasar/Kronos-small \
    --local-dir "$KRONOS_DIR/Kronos-small"

# 下载 Tokenizer
echo ""
echo "Downloading Kronos-Tokenizer-base..."
echo "=========================================="
hf download NeoQuasar/Kronos-Tokenizer-base \
    --local-dir "$KRONOS_DIR/Kronos-Tokenizer-base"

# 克隆 Kronos 仓库获取源码
echo ""
echo "Cloning Kronos repository for source code..."
echo "=========================================="
TEMP_DIR=$(mktemp -d)
git clone --depth 1 https://github.com/shiyu-coder/Kronos.git "$TEMP_DIR/Kronos"

# 复制核心文件
echo ""
echo "Copying Kronos source files..."
echo "=========================================="
if [ -d "$TEMP_DIR/Kronos/kronos" ]; then
    cp "$TEMP_DIR/Kronos/kronos/"*.py "$KRONOS_LIB_DIR/" 2>/dev/null || true
    echo "Copied files from kronos/ directory"
elif [ -d "$TEMP_DIR/Kronos/src" ]; then
    cp "$TEMP_DIR/Kronos/src/"*.py "$KRONOS_LIB_DIR/" 2>/dev/null || true
    echo "Copied files from src/ directory"
else
    # 尝试找到 model.py
    find "$TEMP_DIR/Kronos" -name "model.py" -exec cp {} "$KRONOS_LIB_DIR/" \; 2>/dev/null || true
    find "$TEMP_DIR/Kronos" -name "utils.py" -exec cp {} "$KRONOS_LIB_DIR/" \; 2>/dev/null || true
    echo "Copied model.py and utils.py"
fi

# 清理临时目录
rm -rf "$TEMP_DIR"

# 安装依赖
echo ""
echo "Installing dependencies..."
echo "=========================================="
pip install transformers safetensors einops

# 验证安装
echo ""
echo "=========================================="
echo "Installation Summary"
echo "=========================================="
echo "Model directory: $KRONOS_DIR"
ls -la "$KRONOS_DIR"
echo ""
echo "Library directory: $KRONOS_LIB_DIR"
ls -la "$KRONOS_LIB_DIR"
echo ""
echo "=========================================="
echo "Done! You can now run:"
echo "  freqtrade backtesting -c user_data/config_kronos.json -s KronosStrategy --freqaimodel KronosPredictorModel"
echo "=========================================="
