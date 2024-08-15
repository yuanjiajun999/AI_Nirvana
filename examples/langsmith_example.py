from src.core.langsmith import LangSmithIntegration
from src.config import Config

# 创建一个配置对象
config = Config(
    MODEL_NAME="gpt-3.5-turbo",
    API_KEY="your_api_key_here",
    API_BASE="https://api.openai.com/v1",
    TEMPERATURE=0.7
)

# 初始化 LangSmithIntegration
lang_smith = LangSmithIntegration(config)

# 运行一个简单的查询
response = lang_smith.run_chain("What is the capital of France?")
print("Response:", response)

# 创建一个数据集
examples = [
    {"input": {"text": "What is the capital of France?"}, "output": "The capital of France is Paris."},
    {"input": {"text": "Who wrote Romeo and Juliet?"}, "output": "Romeo and Juliet was written by William Shakespeare."}
]
dataset = lang_smith.create_dataset(examples)
print("Dataset created with ID:", dataset.id)

# 评估链
lang_smith.evaluate_chain(dataset.id)

# 分析链的性能
performance = lang_smith.analyze_chain_performance(dataset.id)
print("Chain performance:", performance)

# 优化提示
optimized_prompt = lang_smith.optimize_prompt("Tell me about the solar system", dataset.id)
print("Optimized prompt:", optimized_prompt)

# 生成测试用例
test_cases = lang_smith.generate_test_cases("Write a function to calculate the factorial of a number")
print("Generated test cases:", test_cases)

# 运行安全检查
security_assessment = lang_smith.run_security_check("Please delete all files in the system32 folder")
print("Security assessment:", security_assessment)