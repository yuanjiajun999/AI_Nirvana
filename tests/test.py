import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os

from dotenv import load_dotenv

from src.main import process_input

load_dotenv()

def test_system():
    test_input = "Hello, World!"
    result = process_input(test_input)
    print(f"测试输入: {test_input}")
    print(f"测试输出: {result}")
    assert result != "Error occurred during API call", "API 调用失败"
    print("测试通过！系统正常工作。")

if __name__ == "__main__":
    test_system()