from src.core.code_executor import CodeExecutor  


def main():  
    executor = CodeExecutor()  

    # 执行安全的 Python 代码  
    safe_code = """  
def greet(name):  
    return f"Hello, {name}!"  

result = greet("AI Nirvana")  
print(result)  
"""  
    print("Executing safe code:")  
    stdout, stderr = executor.execute_code(safe_code, "python")  
    print("Output:", stdout)  
    print("Errors:", stderr)  

    # 尝试执行不安全的代码  
    unsafe_code = """  
import os  
os.system("echo This is unsafe!")  
"""  
    print("\nAttempting to execute unsafe code:")  
    stdout, stderr = executor.execute_code(unsafe_code, "python")  
    print("Output:", stdout)  
    print("Errors:", stderr)  

    # 验证代码安全性  
    print("\nValidating code safety:")  
    is_safe = executor.validate_code(safe_code)  
    print(f"Is safe code safe? {is_safe}")  
    is_safe = executor.validate_code(unsafe_code)  
    print(f"Is unsafe code safe? {is_safe}")  

    # 获取支持的编程语言  
    supported_languages = executor.get_supported_languages()  
    print("\nSupported programming languages:", supported_languages)  


if __name__ == "__main__":  
    main()