#!/bin/bash

# 检查是否安装了 Docker
if ! command -v docker &> /dev/null; then
    echo "Error: Docker 未安装"
    exit 1
fi

# 检查 NVIDIA Container Toolkit
if ! docker info | grep -q "Runtimes.*nvidia"; then
    echo "Warning: 未检测到 NVIDIA Runtime，请确保已安装 nvidia-container-toolkit"
    echo "尝试运行: sudo apt-get install -y nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
fi

echo "=== 开始构建针对 DGX 优化的 Freqtrade 镜像 ==="
docker compose -f docker/docker-compose-dgx.yml build

echo "=== 构建完成 ==="
echo "使用以下命令启动:"
echo "docker compose -f docker/docker-compose-dgx.yml up -d"

