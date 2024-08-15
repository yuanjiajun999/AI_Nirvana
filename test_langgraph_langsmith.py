import logging  # 添加 logging 导入
from src.config import APIConfig, Config  # 直接从 config.py 导入
from src.core.langgraph import LangGraph  # 确保路径正确
from src.core.langsmith import LangSmithIntegration

# 设置日志级别为 DEBUG
logging.basicConfig(level=logging.DEBUG)

def main():
    config = Config()  # 初始化 Config 对象

    try:
        logging.debug("Testing LangGraph...")
        lg = LangGraph()  # 测试 LangGraph 的初始化
        logging.debug("LangGraph test passed.")

        logging.debug("Testing LangSmithIntegration...")
        ls = LangSmithIntegration(config)  # 测试 LangSmithIntegration 的初始化
        logging.debug("LangSmithIntegration test passed.")

    except Exception as e:
        logging.error(f"Test failed: {e}")
        
if __name__ == "__main__":
    main()  # 调用主函数
