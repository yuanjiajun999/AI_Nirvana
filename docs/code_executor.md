CodeExecutor: 安全执行代码的工具
CodeExecutor 是一个 Python 类,用于安全地执行给定编程语言的代码。它提供了以下功能:

安全地执行代码: 使用 execute_code() 方法可以安全地执行 Python 代码,并返回标准输出和标准错误。
验证代码安全性: 使用 validate_code() 方法可以检查给定的代码是否安全。
获取支持的编程语言: 使用 get_supported_languages() 方法可以获取支持的编程语言列表。
使用示例
下面是一个使用 CodeExecutor 的示例程序:

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
API 文档
CodeExecutor
CodeExecutor 类提供了以下方法:

execute_code(code, language)
安全地执行给定编程语言的代码。

参数:

code (str): 要执行的代码
language (str): 编程语言名称
返回值:

tuple: 执行结果的标准输出和标准错误
validate_code(code)
验证给定的代码是否安全。

参数:

code (str): 要验证的代码
返回值:

bool: 如果代码安全,返回 True,否则返回 False
get_supported_languages()
获取支持的编程语言列表。

返回值:

list: 支持的编程语言列表