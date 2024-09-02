# 基础镜像  
FROM python:3.9-slim AS builder  

WORKDIR /app  

# 安装必要的系统依赖  
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*  

# 安装 Poetry  
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir poetry  

# 复制 pyproject.toml 和 poetry.lock  
COPY pyproject.toml poetry.lock ./  

# 创建 Linux 环境特定的 requirements 文件  
RUN poetry export --without-hashes --output requirements.txt \
    && sed -i '/pywin32/d' requirements.txt  

# 最终镜像  
FROM python:3.9-slim  

WORKDIR /app  

# 安装 jq 和其他运行时依赖  
RUN apt-get update && apt-get install -y --no-install-recommends \
    jq \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*  

# 从构建器阶段复制 requirements.txt 并安装依赖  
COPY --from=builder /app/requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt  

# 复制应用程序代码和配置文件  
COPY src ./src  
COPY config.json ./config.json  

# 创建启动脚本  
RUN echo '#!/bin/bash\n\
set -e\n\
if [ -z "$API_KEY" ] || [ -z "$LANGSMITH_API_KEY" ]; then\n\
    echo "Error: API_KEY or LANGSMITH_API_KEY is not set"\n\
    exit 1\n\
fi\n\
jq ".api_key = \$API_KEY | .langsmith_api_key = \$LANGSMITH_API_KEY" config.json > config.tmp.json\n\
mv config.tmp.json config.json\n\
exec python src/main.py\n\
' > start.sh && chmod +x start.sh  

# 设置默认启动命令  
CMD ["./start.sh"]