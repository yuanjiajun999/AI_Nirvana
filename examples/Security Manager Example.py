from src.utils.security import SecurityManager


def main():
    security_manager = SecurityManager()

    # 测试代码安全检查
    safe_code = "print('Hello, World!')"
    unsafe_code = "import os; os.system('rm -rf /')"

    print("Safe code check:", security_manager.is_safe_code(safe_code))
    print("Unsafe code check:", security_manager.is_safe_code(unsafe_code))

    # 测试敏感数据加密和解密
    sensitive_data = "This is sensitive information"
    encrypted_data = security_manager.encrypt_sensitive_data(sensitive_data)
    print("Encrypted data:", encrypted_data)

    decrypted_data = security_manager.decrypt_sensitive_data(encrypted_data)
    print("Decrypted data:", decrypted_data)

    # 测试安全的代码执行
    code_to_execute = """
def greet(name):
    return f"Hello, {name}!"

result = greet("AI Nirvana")
print(result)
"""
    print("\nExecuting code in sandbox:")
    stdout, stderr = security_manager.execute_in_sandbox(code_to_execute, "python")
    print("Stdout:", stdout)
    print("Stderr:", stderr)

    # 测试不安全代码的执行
    unsafe_code_to_execute = """
import os
os.system("echo This should not be executed")
"""
    print("\nAttempting to execute unsafe code:")
    try:
        security_manager.execute_in_sandbox(unsafe_code_to_execute, "python")
    except Exception as e:
        print("Caught exception:", str(e))


if __name__ == "__main__":
    main()
