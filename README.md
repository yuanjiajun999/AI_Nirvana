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
- **模型解释性**：提供对AI决策过程的深入洞察，增加透明度。
- **主动学习**：智能选择最有价值的数据点进行标注，提高数据效率。

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
高级功能
pythonCopyfrom core.reinforcement_learning import ReinforcementLearningAgent
from core.auto_feature_engineering import AutoFeatureEngineer
from core.model_interpretability import ModelInterpreter
from core.active_learning import ActiveLearner

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
CopyOPENAI_API_KEY=your_openai_api_key
API_KEY=your_wildcard_api_key
API_BASE=https://api.gptsapi.net/v1

运行程序：
Copypython src/main.py


安全使用指南

所有用户输入都会经过安全检查，以防止潜在的代码注入攻击。
使用 encrypt_sensitive_data 方法加密敏感信息before存储或传输。
使用 decrypt_sensitive_data 方法解密加密的数据。
定期检查和更新安全规则，以应对新的安全威胁。

项目结构
CopyAI_Nirvana/
├── src/
│   ├── core/
│   │   ├── ai_assistant.py
│   │   ├── code_executor.py
│   │   ├── language_model.py
│   │   ├── reinforcement_learning.py
│   │   ├── auto_feature_engineering.py
│   │   ├── model_interpretability.py
│   │   └── active_learning.py
│   ├── utils/
│   │   ├── security.py
│   │   └── error_handler.py
│   ├── dialogue_manager.py
│   ├── ui.py
│   └── main.py
├── tests/
├── docs/
│   ├── user_guide.md
│   ├── api_reference.md
│   └── developer_guide.md
├── configs/
├── scripts/
├── requirements.txt
├── .env
└── README.md
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