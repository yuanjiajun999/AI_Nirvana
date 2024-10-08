import os
import re
import subprocess
from typing import List, Tuple, Union
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from src.utils.error_handler import SecurityException, error_handler, logger


class SecurityManager:
    """
    安全管理类，提供代码安全检查、敏感数据加密和安全代码执行功能。
    """

    def __init__(self):
        """
        初始化 SecurityManager，生成加密密钥。
        """
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)
        self.unsafe_patterns: List[str] = [
            r"import\s+os",
            r"import\s+subprocess",
            r"open\(",
            r"eval\(",
            r"exec\(",
            r"__import__\(",
            r"globals\(\)",
            r"locals\(\)",
            r"sys\.",
            r"shutil\.",
        ]
        self.password_policy = {
            "min_length": 8,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_digit": True,
            "require_special_char": True,
        }
        logger.info("SecurityManager initialized")

    @error_handler
    def is_safe_code(self, code: str) -> bool:
        """
        检查给定的代码是否安全。

        Args:
            code (str): 要检查的代码字符串。

        Returns:
            bool: 如果代码安全返回 True，否则返回 False。
        """
        unsafe_modules = ['os', 'subprocess', 'sys', 'socket']
        unsafe_functions = ['eval', 'exec', 'compile', '__import__']

        for module in unsafe_modules:
            if f"import {module}" in code or f"from {module}" in code:
                return False

        for func in unsafe_functions:
            if func in code:
                return False

        return True

    @error_handler
    def encrypt_sensitive_data(self, data: Union[str, bytes]) -> str:
        """
        加密敏感数据。

        Args:
            data (Union[str, bytes]): 要加密的数据。

        Returns:
            str: 加密后的数据（Base64 编码的字符串）。

        Raises:
            SecurityException: 如果加密过程中发生错误。
        """
        try:
            if isinstance(data, str):
                data = data.encode()
            encrypted_data = self.cipher_suite.encrypt(data)
            logger.info("Data encrypted successfully")
            return encrypted_data.decode()
        except Exception as e:
            logger.error(f"Encryption error: {str(e)}")
            raise SecurityException("Failed to encrypt data")

    @error_handler
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """
        解密敏感数据。

        Args:
            encrypted_data (str): 要解密的数据（Base64 编码的字符串）。

        Returns:
            str: 解密后的数据。

        Raises:
            SecurityException: 如果解密过程中发生错误。
        """
        try:
            decrypted_data = self.cipher_suite.decrypt(encrypted_data.encode())
            logger.info("Data decrypted successfully")
            return decrypted_data.decode()
        except InvalidToken:
            logger.error("Invalid token: Data may have been tampered with")
            raise SecurityException("Invalid encrypted data")
        except Exception as e:
            logger.error(f"Decryption error: {str(e)}")
            raise SecurityException("Failed to decrypt data")

    @error_handler
    def add_unsafe_pattern(self, pattern: str) -> None:
        """
        添加新的不安全代码模式。

        Args:
            pattern (str): 正则表达式模式。
        """
        self.unsafe_patterns.append(pattern)
        logger.info(f"Added new unsafe pattern: {pattern}")

    @error_handler
    def remove_unsafe_pattern(self, pattern: str) -> None:
        """
        移除不安全代码模式。

        Args:
            pattern (str): 要移除的正则表达式模式。
        """
        if pattern in self.unsafe_patterns:
            self.unsafe_patterns.remove(pattern)
            logger.info(f"Removed unsafe pattern: {pattern}")

    @error_handler
    def execute_in_sandbox(self, code: str, language: str) -> Tuple[str, str]:
        """
        在沙盒环境中执行代码。

        Args:
            code (str): 要执行的代码。
            language (str): 代码的编程语言。

        Returns:
            Tuple[str, str]: (执行结果, 错误信息)

        Raises:
            SecurityException: 如果代码不安全或执行过程中发生错误。
        """
        if not self.is_safe_code(code):
            raise SecurityException("Unsafe code detected")

        sandbox_dir = self.create_sandbox()
        file_path = os.path.join(sandbox_dir, f"script.{language}")
        with open(file_path, "w") as f:
            f.write(code)

        if language == "python":
            cmd = ["python", file_path]
        elif language == "javascript":
            cmd = ["node", file_path]
        else:
            raise SecurityException(f"Unsupported language: {language}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            logger.info(f"Code executed in sandbox: {code[:50]}...")
            return result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            logger.warning("Code execution timed out")
            return "", "Execution timed out"
        except Exception as e:
            logger.error(f"Error executing code: {str(e)}")
            raise SecurityException("Failed to execute code")
        finally:
            os.remove(file_path)

    def create_sandbox(self) -> str:
        """
        创建一个沙盒目录用于执行代码。

        Returns:
            str: 沙盒目录的路径。
        """
        sandbox_dir = os.path.join(os.getcwd(), "sandbox")
        os.makedirs(sandbox_dir, exist_ok=True)
        logger.info(f"Created sandbox directory: {sandbox_dir}")
        return sandbox_dir

    def sanitize_input(self, input_str: str) -> str:
        """
        清理输入字符串，移除潜在的危险字符。

        Args:
            input_str (str): 需要清理的输入字符串。

        Returns:
            str: 清理后的字符串。
        """
        return re.sub(r'[<>&\'"]/g', "", input_str)

    def hash_password(self, password: str) -> str:
        """
        对密码进行哈希处理。

        Args:
            password (str): 需要哈希的密码。

        Returns:
            str: 哈希后的密码。
        """
        return hashlib.sha256(password.encode()).hexdigest()

    def validate_password(self, password: str) -> List[str]:
        """
        验证密码是否符合安全策略。

        Args:
            password (str): 需要验证的密码。

        Returns:
            List[str]: 不符合安全策略的错误信息列表。如果密码符合所有策略，返回空列表。
        """
        errors = []
        if len(password) < self.password_policy["min_length"]:
            errors.append(
                f"Password must be at least {self.password_policy['min_length']} characters long"
            )
        if self.password_policy["require_uppercase"] and not any(
            c.isupper() for c in password
        ):
            errors.append("Password must contain at least one uppercase letter")
        if self.password_policy["require_lowercase"] and not any(
            c.islower() for c in password
        ):
            errors.append("Password must contain at least one lowercase letter")
        if self.password_policy["require_digit"] and not any(
            c.isdigit() for c in password
        ):
            errors.append("Password must contain at least one digit")
        if self.password_policy["require_special_char"] and not re.search(
            r'[!@#$%^&*(),.?":{}|<>]', password
        ):
            errors.append("Password must contain at least one special character")
        return errors
