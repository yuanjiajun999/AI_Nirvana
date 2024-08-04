# AI Nirvana

AI Nirvana 是一个强大的 AI 辅助系统，集成了自然语言处理、代码执行和安全管理功能。

## 主要功能

1. 低秩适应（LoRA）：提高模型训练效率和性能
   - 实现了 LoRALayer 和 LoRAModel，支持高效的模型微调
2. 多模态交互：集成文本、语音、图像等多种交互方式
3. 半监督学习：利用未标记数据进行模型训练
   - 实现了 SemiSupervisedDataset 和 SemiSupervisedTrainer
4. 数字孪生技术：创建物理或虚拟资产的数字模型
5. 隐私增强技术（PETs）：在保护隐私的同时进行数据处理
6. 智能代理系统：能够自主执行复杂任务的AI系统
7. 量化技术：提高模型精度，减少计算需求

8. 智能对话：基于先进的语言模型生成响应
9. **文本摘要**：使用 'summarize' 命令生成文本摘要。
10.**情感分析**：使用 'sentiment' 命令分析文本的情感倾向。
11.**模型切换**：使用 'change_model' 命令切换不同的语言模型。
12.安全管理：提供代码安全检查和敏感数据加密功能
13.**代码执行**：现在可以安全地执行 Python 代  码。使用 'execute' 命令来尝试。
14.对话历史管理：保存和清除对话历史
15.LangChain 集成：支持高级语言模型链和智能代理
16.LangGraph 支持：实现基于图的语言处理
17.LangSmith 集成：提供语言模型评估和优化工具

## 增安全特性

- 代码安全检查：自动检测并阻止潜在的危险代码执行
- 敏感数据加密：使用强加密算法保护敏感信息
- 动态安全规则：支持添加和移除自定义的安全检查规则

## API Integration

This project now uses the latest version of LangChain with OpenAI integration. To use the API:

1. Set the following environment variables:
   - `API_KEY`: Your API key
   - `API_BASE`: The base URL for the API (default: https://api.gptsapi.net/v1)

2. Use the `get_response` function from `src.core.langchain` to interact with the API:

   ```python
   from src.core.langchain import get_response

   response = get_response("Your question here")
   print(response)
新增功能

强化学习：通过与环境交互来优化决策过程。
自动特征工程：自动发现和创建有助于提高模型性能的特征。
模型解释性：提供对 AI 决策过程的深入洞察，增加透明度。
主动学习：智能选择最有价值的数据点进行标注，提高数据效率。
LangChain 集成：提供强大的语言模型链和智能代理功能
LangGraph 支持：实现基于图的知识检索和推理
LangSmith 工具：用于代码生成、重构和文本翻译

使用示例
基本功能
pythonCopyfrom src.core.ai_assistant import AIAssistant

assistant = AIAssistant()

# 生成响应
response = assistant.generate_response("Tell me a joke")
print(response)

# 文本摘要
summary = assistant.summarize("Long text here...")
print(summary)

# 情感分析
sentiment = assistant.analyze_sentiment("I love this product!")
print(sentiment)

# 更改模型
assistant.change_model("gpt-4")

# 加密敏感数据
encrypted = assistant.encrypt_sensitive_data("sensitive info")
decrypted = assistant.decrypt_sensitive_data(encrypted)
高级功能
pythonCopyfrom src.core.reinforcement_learning import ReinforcementLearningAgent
from src.core.auto_feature_engineering import AutoFeatureEngineer
from src.core.model_interpretability import ModelInterpreter
from src.core.active_learning import ActiveLearner
from src.core.langchain import LangChainAgent
from src.core.langgraph import LangGraph
from src.core.langsmith import LangSmith

# 使用强化学习
rl_agent = ReinforcementLearningAgent(state_size=10, action_size=5)
action = rl_agent.act(state)

# 使用自动特征工程
auto_fe = AutoFeatureEngineer(data)
feature_matrix, feature_defs = auto_fe.generate_features()

# 使用模型解释性
interpreter = ModelInterpreter(model, X)
interpreter.plot_summary()

# 使用主动学习
active_learner = ActiveLearner(X_pool, y_pool, X_test, y_test)
final_accuracy = active_learner.active_learning_loop(initial_samples=100, n_iterations=10, samples_per_iteration=10)

# 使用LangChain
lang_chain_agent = LangChainAgent()
qa_result = lang_chain_agent.run_qa_task("What is the capital of France?")

# 使用LangGraph
lang_graph = LangGraph()
knowledge = lang_graph.retrieve_knowledge("Who invented the telephone?")

# 使用LangSmith
lang_smith = LangSmith()
generated_code = lang_smith.generate_code("Write a Python function to sort a list")
快速开始

克隆仓库：
Copygit clone https://github.com/yuanjiajun999/AI_Nirvana.git
cd AI_Nirvana

设置环境：
Copypython -m venv venv
source venv/bin/activate  # Windows 使用: venv\Scripts\activate
pip install -r requirements.txt

配置 API 密钥：
创建 .env 文件并添加：
CopyAPI_KEY=your_wildcard_api_key
API_BASE=https://api.gptsapi.net/v1

运行程序：
Copypython src/main.py

使用方法

1. 启动程序后，输入 'help' 查看可用命令。
2. 使用 'execute' 命令可以执行 Python 代码。
3. 使用 'clear' 命令可以清除对话历史。
4. 使用 'sentiment' 命令可以进行情感分析。
5. 使用 'summarize' 命令可以生成文本摘要。
6. 使用 'change_model' 命令可以切换不同的语言模型。
7. 使用 'quit' 命令可以退出程序。
8. 运行测试：python -m pytest

##示例

“examples”目录包含演示如何使用AI Nirvana各种模块的示例脚本。这些示例为使用系统的功能提供了一个起点。

有关可用示例以及如何使用它们的更多信息，请参阅[示例文档]（docs/examples.md）。

## 文档

详细的文档可以在 `docs` 目录中找到：

- [用户指南](docs/user_guide.md)
- [API 参考](docs/api_reference.md)
- [开发者指南](docs/developer_guide.md)

安全注意事项

- 所有用户输入都会经过安全检查，以防止潜在的代码注入攻击。
- 'execute' 命令在安全的沙盒环境中执行 Python 代码。

## 环境设置

   1. 复制 `.env.example` 文件并重命名为 `.env`
   2. 在 `.env` 文件中填入您的实际 API 密钥和其他敏感信息
   3. 运行 Docker 容器时，使用以下命令注入环境变量：

   ```bash
   docker run -p 8000:8000 \
     -e API_KEY=your_actual_api_key \
     -e API_BASE=your_actual_api_base \
     -e LANGSMITH_API_KEY=your_actual_langsmith_key \
     ai_nirvana:latest
   ```

   注意：请不要将您的实际 API 密钥提交到版本控制系统中。
   
## 开发环境设置

为了确保一致的代码质量和开发体验，请按照以下步骤设置您的开发环境：

### 1. Python 环境

确保您已安装 Python 3.9 或更高版本。您可以通过以下命令检查 Python 版本：

```bash
python --version
2. 安装依赖
在项目根目录下运行以下命令安装所需的依赖：
pip install -r requirements.txt
3. 安装 pre-commit
我们使用 pre-commit 来自动化代码检查和格式化。安装 pre-commit：
pip install pre-commit
然后，在项目根目录下运行：
pre-commit install
4. 设置 Git
确保您的 Git 配置正确设置了用户名和邮箱：
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
5. 处理可能的 SSL 问题
如果您在运行 pre-commit 或其他涉及网络连接的操作时遇到 SSL 相关错误，请尝试以下步骤：

更新 pip 和 setuptools：
python -m pip install --upgrade pip setuptools

确保 SSL 模块正确安装：
python -c "import ssl; print(ssl.OPENSSL_VERSION)"

如果上述命令失败，考虑重新安装 Python，确保包含 SSL 支持。
如果您在公司网络中，可能需要配置代理设置：
git config --global http.proxy http://proxyserver:port


6. 代码格式化
我们使用 Black 来保持代码格式的一致性。在提交代码之前，请运行：
black .
7. 提交代码
正常情况下，直接使用 git commit 即可。pre-commit 钩子会自动运行检查。
如果遇到问题，可以使用 --no-verify 标志跳过 pre-commit 钩子：
git commit -m "Your commit message" --no-verify
但请注意，这应该只是临时解决方案，长期应该解决 pre-commit 的问题。
遇到问题？
如果您在设置环境或使用工具时遇到任何问题，请查看项目 wiki 或联系项目维护者寻求帮助。


VSCode 开发环境和 Docker 部署：

VSCode 开发环境设置

将以下内容添加到您的 README.md 文件中：
markdownCopy## 开发环境设置 (VSCode)

我们推荐使用 Visual Studio Code 进行开发。请按照以下步骤设置您的开发环境：

### 1. 安装 VSCode

如果您还没有安装 VSCode，请从 [官方网站](https://code.visualstudio.com/) 下载并安装。

### 2. 安装推荐的 VSCode 扩展

我们建议安装以下 VSCode 扩展来提升开发体验：

- Python
- Pylance
- Black Formatter
- GitLens
- Docker (如果需要在 VSCode 中管理 Docker)

您可以在 VSCode 中通过 Ctrl+Shift+X (Windows/Linux) 或 Cmd+Shift+X (Mac) 打开扩展视图，然后搜索并安装这些扩展。

### 3. 配置 Python 环境

1. 打开项目文件夹在 VSCode 中。
2. 创建一个虚拟环境（如果还没有的话）：
python -m venv venv
3. 在 VSCode 中选择 Python 解释器：
- 按 `Ctrl+Shift+P` (Windows/Linux) 或 `Cmd+Shift+P` (Mac)
- 输入 "Python: Select Interpreter"
- 选择刚创建的虚拟环境

### 4. 安装依赖

在 VSCode 的终端中运行：
pip install -r requirements.txt

### 5. 设置 pre-commit

在 VSCode 的终端中运行：
pip install pre-commit
pre-commit install

### 6. 配置 Black 格式化

1. 打开 VSCode 设置 (File > Preferences > Settings)
2. 搜索 "Python Formatting Provider"
3. 选择 "black" 作为格式化器
4. 启用 "Format On Save" 选项

现在，每次保存 Python 文件时，VSCode 都会自动使用 Black 进行格式化。

### 7. Git 配置

确保在 VSCode 的终端中设置了 Git 用户名和邮箱：
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

## Docker 部署说明

虽然我们主要使用 VSCode 进行开发，但项目的部署仍然使用 Docker。以下是部署相关的说明：

1. 构建 Docker 镜像：
docker build -t project-name .

2. 运行 Docker 容器：
docker run -d -p 8080:8080 project-name



安全使用指南

所有用户输入都会经过安全检查，以防止潜在的代码注入攻击。
使用 encrypt_sensitive_data 方法加密敏感信息 before 存储或传输。
使用 decrypt_sensitive_data 方法解密加密的数据。
定期检查和更新安全规则，以应对新的安全威胁。

项目结构
CopyAI_Nirvana/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── ai_assistant.py
│   │   ├── active_learning.py
│   │   ├── auto_feature_engineering.py
│   │   ├── digital_twin.py
│   │   ├── generative_ai.py
│   │   ├── intelligent_agent.py
│   │   ├── language_model.py
│   │   ├── lora.py
│   │   ├── model_interpretability.py
│   │   ├── multimodal.py
│   │   ├── privacy_enhancement.py
│   │   ├── quantization.py
│   │   ├── reinforcement_learning.py
│   │   ├── semi_supervised_learning.py
│   │   ├── api_model.py
│   │   ├── code_executor.py
│   │   ├── knowledge_base.py
│   │   ├── local_model.py
│   │   ├── optimization.py
│   │   ├── task_planner.py
│   │   ├── langchain.py
│   │   ├── langgraph.py
│   │   └── langsmith.py
│   ├── interfaces/
│   │   ├── api.py
│   │   ├── cli.py
│   │   ├── gui.py
│   │   ├── sd_web_controller.py
│   │   └── voice_interface.py
│   ├── plugins/
│   │   ├── plugin_manager.py
│   │   └── translator.py
│   ├── utils/
│   │   ├── cache_manager.py
│   │   ├── error_handler.py
│   │   ├── logger.py
│   │   └── security.py
│   ├── __init__.py
│   ├── config.py
│   ├── dialogue_manager.py
│   ├── main.py
│   ├── setup.py
│   └── ui.py
├── tests/
│   ├── __init__.py
│   ├── test_digital_twin.py
│   ├── test_generative_ai.py
│   ├── test_intelligent_agent.py
│   ├── test_langchain.py
│   ├── test_langgraph.py
│   ├── test_langsmith.py
│   ├── test_lora.py
│   ├── test_models.py
│   ├── test_multimodal.py
│   ├── test_privacy_enhancement.py
│   ├── test_quantization.py
│   ├── test_semi_supervised_learning.py
│   ├── test_wildcard_api.py
│   └── test.py
├── docs/
│   ├── user_guide.md
│   ├── api_reference.md
│   ├── developer_guide.md
│   └── index.md
├── examples/
│   └── advanced_features_demo.py
├── sandbox/
│   └── script.python
├── scripts/
│   ├── install_requirements.py
│   └── setup.sh
├── .env.example
├── .gitattributes
├── .gitignore
├── README.md
├── requirements.txt
└── LICENSE
详细文档

完整用户指南
API 参考
开发者文档

运行测试
Copypytest tests/
贡献指南

Fork 该仓库
创建您的特性分支 (git checkout -b feature/AmazingFeature)
提交您的更改 (git commit -m 'Add some AmazingFeature')
推送到分支 (git push origin feature/AmazingFeature)
打开一个 Pull Request

许可证
该项目采用 MIT 许可证 - 查看 LICENSE 文件了解详情
## 版本历史

- v1.0.0: 初始版本
- v1.1.0: 添加 LoRA 和半监督学习功能
- v1.2.0: 集成 LangChain、LangGraph 和 LangSmith

依赖管理和 Docker 环境设置指南
依赖文件说明

requirements.txt: 主要依赖文件，包含所有依赖项。
requirements-common.txt: 包含所有平台通用的依赖项。
requirements-docker.txt: Docker 环境专用的依赖文件（由 Dockerfile 自动生成）。

更新依赖

更新 requirements.txt:

使用 pip freeze > requirements.txt 或手动编辑以更新依赖。


更新 requirements-common.txt:

运行 python update_requirements.py 脚本以自动更新通用依赖。



Docker 环境设置
Dockerfile.base
dockerfileCopyFROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-common.txt ./

RUN grep -v -E "pywin32|pywinpty|win32|wincertstore" requirements.txt > requirements-docker.txt && \
    cat requirements-common.txt >> requirements-docker.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt

RUN apt-get purge -y --auto-remove gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV DOCKER_ENV=1
ENV CUDA_VISIBLE_DEVICES=""
Dockerfile
dockerfileCopyFROM ai_nirvana_base:latest

WORKDIR /app

COPY . .

RUN if [ -f requirements-project.txt ]; then \
        pip install --no-cache-dir -r requirements-project.txt; \
    fi

CMD ["python", "src/main.py"]

HEALTHCHECK CMD python -c "import requests; requests.get('http://localhost:8000')"
使用说明
对于开发者

Windows 开发者：使用 requirements.txt
macOS 开发者：使用 requirements.txt
Linux 开发者：使用 requirements.txt

使用 Docker

构建基础镜像：
Copydocker build -t ai_nirvana_base:latest -f Dockerfile.base .

构建应用镜像：
Copydocker build -t ai_nirvana_app:latest .

运行容器：
Copydocker run -it ai_nirvana_app:latest

### 运行测试

本项目使用 `run_docker_tests.py` 脚本运行测试。

### 在 Docker 容器外运行测试

如果您在本地机器上（不在 Docker 容器内），使用以下命令：

1. 构建 Docker 镜像（如果尚未构建）：
docker build -t ai_nirvana_app .

2. 运行测试：
docker run ai_nirvana_app python run_docker_tests.py

### 在 Docker 容器内运行测试

如果您已经在 Docker 容器内，直接运行：
python run_docker_tests.py

注意：Docker 环境测试仅包括与 Linux 环境兼容的测试。Windows 特定的测试将被排除。

代码审查：
检查您的代码，确保核心功能不依赖于 Windows 特定的模块。如果有依赖，考虑使用条件导入或跨平台替代方案。
持续集成：
在 CI 流程中，为不同的环境设置不同的测试任务：

Windows runners 运行完整测试套件
Linux/Docker runners 运行 Docker 兼容的测试


依赖管理：
再次检查 requirements-docker.txt，确保它不包含任何 Windows 特定的包。


注意事项

定期运行 update_requirements.py 以保持 requirements-common.txt 更新。
在提交代码前，确保所有依赖文件都已更新并添加到版本控制中。
如遇到平台特定的依赖问题，请在 issue 中报告。

联系方式
项目链接：https://github.com/yuanjiajun999/AI_Nirvana