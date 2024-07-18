# AI Nirvana

AI Nirvana 是一个强大的 AI 辅助系统，集成了自然语言处理、代码执行和安全管理功能。

## 主要功能

- 智能对话：基于先进的语言模型生成响应
- 文本摘要：自动生成长文本的简洁摘要
- 情感分析：分析文本的情感倾向
- 多模型支持：支持多种 OpenAI 模型，可动态切换
- 安全管理：提供代码安全检查和敏感数据加密功能
- 缓存系统：提高响应速度，减少 API 调用
- 代码执行：安全地执行 Python 代码

## 新增安全特性

- 代码安全检查：自动检测并阻止潜在的危险代码执行
- 敏感数据加密：使用强加密算法保护敏感信息
- 动态安全规则：支持添加和移除自定义的安全检查规则

## 新增功能

- **强化学习**：通过与环境交互来优化决策过程。
- **自动特征工程**：自动发现和创建有助于提高模型性能的特征。
- **模型解释性**：提供对 AI 决策过程的深入洞察，增加透明度。
- **主动学习**：智能选择最有价值的数据点进行标注，提高数据效率。
- **LangChain 集成**：提供强大的语言模型链和智能代理功能
- **LangGraph 支持**：实现基于图的知识检索和推理
- **LangSmith 工具**：用于代码生成、重构和文本翻译

## 使用示例

### 基本功能

```python
from src.core.ai_assistant import AIAssistant

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
```

### 高级功能

```python
from src.core.reinforcement_learning import ReinforcementLearningAgent
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
```

## 快速开始

1. 克隆仓库：
   ```
   git clone https://github.com/yuanjiajun999/AI_Nirvana.git
   cd AI_Nirvana
   ```

2. 设置环境：
   ```
   python -m venv venv
   source venv/bin/activate  # Windows 使用: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. 配置 API 密钥：
   创建 .env 文件并添加：
   ```
   OPENAI_API_KEY=your_openai_api_key
   API_KEY=your_wildcard_api_key
   API_BASE=https://api.gptsapi.net/v1
   ```

4. 运行程序：
   ```
   python src/main.py
   ```

## 安全使用指南

- 所有用户输入都会经过安全检查，以防止潜在的代码注入攻击。
- 使用 `encrypt_sensitive_data` 方法加密敏感信息 before 存储或传输。
- 使用 `decrypt_sensitive_data` 方法解密加密的数据。
- 定期检查和更新安全规则，以应对新的安全威胁。

## 项目结构

```
AI_Nirvana/
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
├── requirements-actual.txt
└── requirements.txt
```
## 项目结构

```
AI_Nirvana/
├── src/                      # 源代码目录
│   ├── core/                 # 核心功能模块
│   ├── interfaces/           # 用户接口模块
│   ├── plugins/              # 插件模块
│   ├── utils/                # 工具函数和类
│   ├── config.py             # 配置文件
│   ├── dialogue_manager.py   # 对话管理器
│   ├── main.py               # 主程序入口
│   └── ui.py                 # 用户界面
├── tests/                    # 测试目录
├── docs/                     # 文档目录
├── examples/                 # 示例代码
├── sandbox/                  # 沙盒环境
├── scripts/                  # 脚本文件
├── requirements.txt          # 项目依赖
└── README.md                 # 项目说明文档
```

详细的项目结构可以在项目根目录下查看。

## 详细文档

- [完整用户指南](docs/user_guide.md)
- [API 参考](docs/api_reference.md)
- [开发者文档](docs/developer_guide.md)

## 运行测试

```
pytest tests/
```

## 贡献指南

1. Fork 该仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request

## 许可证

该项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

项目链接：https://github.com/yuanjiajun999/AI_Nirvana