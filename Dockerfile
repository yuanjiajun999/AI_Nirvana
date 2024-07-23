FROM ai_nirvana_base:latest

WORKDIR /app

COPY . .
COPY pytest.ini .
COPY run_docker_tests.py .

# 安装任何额外的项目特定依赖（如果有的话）
RUN if [ -f requirements-project.txt ]; then \
        pip install --no-cache-dir -r requirements-project.txt; \
    fi

# 验证 PyTorch 安装
RUN python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

CMD ["python", "src/main.py"]

HEALTHCHECK CMD python -c "import requests; requests.get('http://localhost:8000')"