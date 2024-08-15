FROM ai_nirvana_base:latest  

WORKDIR /app  

# 安装 Poetry  
RUN pip install --no-cache-dir poetry  

# 复制 Poetry 相关文件  
COPY pyproject.toml poetry.lock ./  

# 安装依赖  
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi  

# 复制其他项目文件  
COPY . .  

# 验证 PyTorch 安装  
RUN python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"  

CMD ["python", "src/main.py"]  

HEALTHCHECK CMD python -c "import requests; requests.get('http://localhost:8000')"