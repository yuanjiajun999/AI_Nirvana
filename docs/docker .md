# AI Nirvana Docker 使用文档

## 目录
1. [项目介绍](#项目介绍)
2. [环境要求](#环境要求)
3. [项目结构](#项目结构)
4. [快速开始](#快速开始)
5. [开发工作流程](#开发工作流程)
6. [生产环境部署](#生产环境部署)
7. [问题排查](#问题排查)
8. [最佳实践](#最佳实践)
9. [常见问题](#常见问题)

## 1. 项目介绍

AI Nirvana 是一个强大的 AI 助手项目，使用 Docker 容器化以便于部署和开发。本文档提供了使用 Docker 设置、开发和部署 AI Nirvana 项目的全面指导。

## 2. 环境要求

- Docker（版本 20.10 或更高）
- Docker Compose（版本 1.29 或更高）
- Git
- （可选）Poetry 用于本地开发

## 3. 项目结构

```
AI_Nirvana-1/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── commands.py
│   ├── config.py
│   └── ...
├── tests/
├── docker/
│   ├── base.Dockerfile
│   └── dev.Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── poetry.lock
└── README.md
```

## 4. 快速开始

### 克隆仓库：
```bash
git clone https://github.com/your-repo/AI_Nirvana-1.git
cd AI_Nirvana-1
```

### 构建并运行 Docker 容器：
```bash
docker-compose up --build -d
```

此命令将构建 Docker 镜像并在后台启动容器。

### 验证应用是否正在运行：
```bash
docker-compose ps
```

您应该能看到 `app` 服务正在运行。

### 访问应用：
现在应该可以通过 `http://localhost:8000` 访问应用（或 `docker-compose.yml` 中指定的端口）。

## 5. 开发工作流程

### 修改代码：
1. 在本地环境中修改代码。
2. 由于卷挂载，更改将直接反映在运行的容器中。

### 添加新依赖：
1. 在 `pyproject.toml` 中添加依赖：
   ```toml
   [tool.poetry.dependencies]
   new_package = "^1.0.0"
   ```
2. 更新锁定文件：
   ```bash
   poetry lock
   ```
3. 重新构建 Docker 镜像：
   ```bash
   docker-compose build
   ```
4. 重启容器：
   ```bash
   docker-compose up -d
   ```

### 运行测试：
```bash
docker-compose run --rm app pytest
```

### 进入容器shell：
```bash
docker exec -it ai_nirvana-1-app-1 /bin/bash
```

## 6. 生产环境部署

对于生产环境部署，使用 `prod.Dockerfile`：

1. 构建生产镜像：
   ```bash
   docker build -f docker/prod.Dockerfile -t ai-nirvana-prod:latest .
   ```

2. 运行生产容器：
   ```bash
   docker run -d -p 8000:8000 --name ai-nirvana-prod ai-nirvana-prod:latest
   ```

## 7. 问题排查

### 常见问题：

1. **容器无法启动：**
   - 检查日志：`docker-compose logs app`
   - 确保所有必要的环境变量已设置

2. **依赖问题：**
   - 重新构建镜像：`docker-compose build --no-cache`
   - 检查 `pyproject.toml` 和 `poetry.lock` 是否有冲突

3. **端口冲突：**
   - 在 `docker-compose.yml` 中更改端口映射

## 8. 最佳实践

1. 始终对 Dockerfile 和 docker-compose.yml 使用版本控制。
2. 保持基础镜像更新。
3. 使用多阶段构建以最小化最终镜像大小。
4. 不要在 Docker 镜像中存储机密信息。
5. 使用 `.dockerignore` 排除不必要的文件。

## 9. 常见问题

问：如何在生产环境中更新应用？
答：重新构建生产镜像，停止旧容器，然后使用更新后的镜像启动新容器。

问：我可以使用 Docker 进行本地开发吗？
答：是的，`dev.Dockerfile` 和 `docker-compose.yml` 已配置为支持本地开发，包括热重载功能。

问：如何添加自定义环境变量？
答：在 `docker-compose.yml` 的 `environment` 部分添加，或使用 `.env` 文件。

如有更多问题或需要支持，请在项目的 GitHub 仓库中开 issue。


# AI Nirvana Docker Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Project Structure](#project-structure)
4. [Getting Started](#getting-started)
5. [Development Workflow](#development-workflow)
6. [Production Deployment](#production-deployment)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)
9. [FAQs](#faqs)

## 1. Introduction

AI Nirvana is a powerful AI assistant project containerized using Docker for easy deployment and development. This document provides comprehensive instructions for setting up, developing, and deploying the AI Nirvana project using Docker.

## 2. Prerequisites

- Docker (version 20.10 or later)
- Docker Compose (version 1.29 or later)
- Git
- (Optional) Poetry for local development

## 3. Project Structure

```
AI_Nirvana-1/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── commands.py
│   ├── config.py
│   └── ...
├── tests/
├── docker/
│   ├── base.Dockerfile
│   └── dev.Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── poetry.lock
└── README.md
```

## 4. Getting Started

### Clone the repository:
```bash
git clone https://github.com/your-repo/AI_Nirvana-1.git
cd AI_Nirvana-1
```

### Build and run the Docker containers:
```bash
docker-compose up --build -d
```

This command builds the Docker images and starts the containers in detached mode.

### Verify the application is running:
```bash
docker-compose ps
```

You should see the `app` service running.

### Access the application:
The application should now be accessible at `http://localhost:8000` (or the port specified in your `docker-compose.yml`).

## 5. Development Workflow

### Modifying the codebase:
1. Make changes to the code in your local environment.
2. The changes will be reflected in the running container due to volume mounting.

### Adding new dependencies:
1. Add the dependency to `pyproject.toml`:
   ```toml
   [tool.poetry.dependencies]
   new_package = "^1.0.0"
   ```
2. Update the lock file:
   ```bash
   poetry lock
   ```
3. Rebuild the Docker image:
   ```bash
   docker-compose build
   ```
4. Restart the containers:
   ```bash
   docker-compose up -d
   ```

### Running tests:
```bash
docker-compose run --rm app pytest
```

### Accessing the container shell:
```bash
docker exec -it ai_nirvana-1-app-1 /bin/bash
```

## 6. Production Deployment

For production deployment, use the `prod.Dockerfile`:

1. Build the production image:
   ```bash
   docker build -f docker/prod.Dockerfile -t ai-nirvana-prod:latest .
   ```

2. Run the production container:
   ```bash
   docker run -d -p 8000:8000 --name ai-nirvana-prod ai-nirvana-prod:latest
   ```

## 7. Troubleshooting

### Common Issues:

1. **Container fails to start:**
   - Check logs: `docker-compose logs app`
   - Ensure all required environment variables are set

2. **Dependencies issues:**
   - Rebuild the image: `docker-compose build --no-cache`
   - Check `pyproject.toml` and `poetry.lock` for conflicts

3. **Port conflicts:**
   - Change the port mapping in `docker-compose.yml`

## 8. Best Practices

1. Always use version control for your Dockerfiles and docker-compose.yml.
2. Keep the base image updated.
3. Use multi-stage builds to minimize final image size.
4. Don't store secrets in Docker images.
5. Use `.dockerignore` to exclude unnecessary files.

## 9. FAQs

Q: How do I update the application in production?
A: Rebuild the production image, stop the old container, and start a new one with the updated image.

Q: Can I use Docker for local development?
A: Yes, the `dev.Dockerfile` and `docker-compose.yml` are configured for local development with hot-reloading.

Q: How do I add custom environment variables?
A: Add them to the `environment` section in `docker-compose.yml` or use a `.env` file.

For more questions or support, please open an issue on the project's GitHub repository.