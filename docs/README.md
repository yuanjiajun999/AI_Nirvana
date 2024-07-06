# AI Nirvana

AI Nirvana 是一个强大的 AI 辅助系统，集成了自然语言处理、代码执行和安全管理功能。

## 项目结构
AI_Nirvana/
├── src/
│   ├── core/
│   │   ├── code_executor.py
│   │   └── language_model.py
│   ├── utils/
│   │   └── security.py
│   └── main.py
├── tests/
│   ├── test_code_executor.py
│   ├── test_language_model.py
│   ├── test_security.py
│   └── test.py
├── docs/
│   └── README.md
├── configs/
│   └── pyproject.toml
├── scripts/
│   ├── setup.sh
│   └── install_requirements.py
├── .gitignore
├── requirements.txt
└── .env

## 设置说明

1. 克隆仓库：
git clone https://github.com/yuanjiajun999/AI_Nirvana.git 
cd AI_Nirvana

2. 创建虚拟环境：
 python -m venv venv source venv/bin/activate
 # 在 Windows 上使用 venv\Scripts\activate

3. 安装依赖：
pip 安装 -r 要求.txt

4. 设置环境变量：
创建一个 `.env` 文件在项目根目录，并添加以下内容：
 OPENAI_API_KEY=你的openai_api_key
 API_KEY=你的wildcard_api_key 
 API_BASE= https://api.gptsapi.net/v1
复制请确保用您的实际 API 密钥替换相应的值。

## 运行程序

运行主程序：
python src/main.py

## 运行测试

执行测试套件：
pytest 测试/

## 主要功能

- 自然语言处理：使用先进的语言模型生成响应。
- 代码执行：安全地执行 Python 代码。
- 安全管理：提供代码安全检查和敏感数据加密功能。

## 贡献指南

1. Fork 该仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request

## 许可证

该项目采用 MIT 许可证 - 查看 [LICENSE.md](LICENSE.md) 文件了解详情

## 联系方式

项目链接：[https://github.com/yuanjiajun999/AI_Nirvana](https://github.com/yuanjiajun999/AI_Nirvana)