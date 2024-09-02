# 基于基础镜像
FROM your-project-base:latest

# 设置工作目录
WORKDIR /app

# 确保最新的代码被复制（CI/CD 中执行）
COPY app ./app

# 暴露应用运行端口
EXPOSE 8000

# 设置生产环境启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
