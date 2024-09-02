# 基础镜像使用官方的 Python 3.9-slim
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统级依赖（如有必要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 Poetry
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir poetry

# 将 pyproject.toml 和 poetry.lock 复制到容器
COPY pyproject.toml poetry.lock ./

# 配置 Poetry，不创建虚拟环境
RUN poetry config virtualenvs.create false

# 安装生产依赖
RUN poetry install --no-dev --no-interaction --no-ansi

# 设置默认启动命令（可在具体环境的 Dockerfile 中覆盖）
CMD ["python", "app/main.py"]
