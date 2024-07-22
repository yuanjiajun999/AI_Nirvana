#!/bin/bash

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 检查 .env 文件是否存在
if [ ! -f .env ]; then
    echo "请创建 .env 文件并添加您的 API 密钥"
    echo "API_KEY=your_api_key_here" > .env
    echo "API_BASE=https://api.gptsapi.net/v1" >> .env
fi

echo "设置完成。请运行 'source venv/bin/activate' 来激活虚拟环境。"
