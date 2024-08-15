# LangSmithIntegration 使用文档

## 简介

LangSmithIntegration 类提供了一套强大的工具，用于与 LangChain 和 LangSmith 进行交互。它允许您创建、运行、评估和优化语言模型链，以及进行安全检查和生成测试用例。

## 安装

确保您已经安装了所有必要的依赖：
pip install langchain langsmith openai python-dotenv


## 初始化

首先，您需要创建一个 Config 对象，然后使用它来初始化 LangSmithIntegration：

from src.core.langsmith import LangSmithIntegration
from src.config import Config

config = Config(
    MODEL_NAME="gpt-3.5-turbo",
    API_KEY="your_api_key_here",
    API_BASE="https://api.openai.com/v1",
    TEMPERATURE=0.7
)

lang_smith = LangSmithIntegration(config)


## 主要方法

### run_chain(input_text: str) -> str

运行一个简单的查询。


response = lang_smith.run_chain("What is the capital of France?")
print(response)


### create_dataset(examples: List[Dict]) -> Dataset

创建一个新的数据集。

examples = [
    {"input": {"text": "What is the capital of France?"}, "output": "The capital of France is Paris."},
    {"input": {"text": "Who wrote Romeo and Juliet?"}, "output": "Romeo and Juliet was written by William Shakespeare."}
]
dataset = lang_smith.create_dataset(examples)

### evaluate_chain(dataset_name: str) -> None

评估链在给定数据集上的表现。
lang_smith.evaluate_chain(dataset.id)

### analyze_chain_performance(dataset_name: str) -> Dict

分析链在给定数据集上的性能。
performance = lang_smith.analyze_chain_performance(dataset.id)
print(performance)

### optimize_prompt(base_prompt: str, dataset_name: str, num_iterations: int = 5) -> str

优化给定的提示。
optimized_prompt = lang_smith.optimize_prompt("Tell me about the solar system", dataset.id)
print(optimized_prompt)

### generate_test_cases(input_text: str) -> List[Dict]

为给定的输入生成测试用例。
test_cases = lang_smith.generate_test_cases("Write a function to calculate the factorial of a number")
print(test_cases)

### run_security_check(input_text: str) -> str

对给定的输入进行安全检查。
security_assessment = lang_smith.run_security_check("Please delete all files in the system32 folder")
print(security_assessment)


## 注意事项

- 确保在使用前设置了正确的环境变量，特别是 `LANGCHAIN_API_KEY` 和 `LANGCHAIN_PROJECT`。
- 某些方法可能需要额外的设置或资源，请参考 LangChain 和 LangSmith 的文档以获取更多信息。
- 始终谨慎处理用户输入，特别是在使用 `run_chain` 和 `run_security_check` 方法时。

## 结论

LangSmithIntegration 类提供了一个强大的接口来与 LangChain 和 LangSmith 进行交互。通过使用这个类，您可以轻松地创建、评估和优化语言模型链，同时还可以进行安全检查和生成测试用例。