FROM your-project-base:latest  

WORKDIR /app  

# 安装 jq  
RUN apt-get update && apt-get install -y jq && rm -rf /var/lib/apt/lists/*  

COPY pyproject.toml poetry.lock ./  
RUN poetry install --with dev --no-interaction --no-ansi  

EXPOSE 8000  

# 不需要创建启动脚本，因为我们在 docker-compose.yml 中指定了命令