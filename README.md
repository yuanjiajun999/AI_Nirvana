# AI Nirvana

AI Nirvana 是一个强大的 AI 辅助系统，集成了自然语言处理、代码执行和安全管理功能。

## 主要功能

- 智能对话：基于先进的语言模型生成响应
- **文本摘要**：使用 'summarize' 命令生成文本摘要。
- **情感分析**：使用 'sentiment' 命令分析文本的情感倾向。
- **模型切换**：使用 'change_model' 命令切换不同的语言模型。
- 安全管理：提供代码安全检查和敏感数据加密功能
- **代码执行**：现在可以安全地执行 Python 代  码。使用 'execute' 命令来尝试。
- 对话历史管理：保存和清除对话历史

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


安全注意事项

- 所有用户输入都会经过安全检查，以防止潜在的代码注入攻击。
- 'execute' 命令在安全的沙盒环境中执行 Python 代码。


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
联系方式
项目链接：https://github.com/yuanjiajun999/AI_Nirvana