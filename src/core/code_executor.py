import subprocess
import tempfile
import os
from typing import Tuple
from src.utils.security import SecurityManager
from src.utils.error_handler import error_handler, logger, AIAssistantException

class CodeExecutor:
    def __init__(self):
        self.security_manager = SecurityManager()
        logger.info("CodeExecutor initialized")

    @error_handler
    def execute_code(self, code: str, language: str) -> Tuple[str, str]:
        """
        执行给定的代码。

        Args:
            code (str): 要执行的代码
            language (str): 代码的编程语言

        Returns:
            Tuple[str, str]: 包含执行结果和错误信息的元组

        Raises:
            AIAssistantException: 如果代码执行失败或不安全
        """
        if language.lower() != 'python':
            logger.warning(f"Unsupported language: {language}")
            return "", "Only Python execution is currently supported."
        
        if not self.security_manager.is_safe_code(code):
            logger.warning("Unsafe code detected")
            return "", "Code execution blocked due to security concerns."

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        try:
            result = subprocess.run(['python', temp_file_path],
                                    capture_output=True,
                                    text=True,
                                    timeout=5)
            logger.info(f"Code executed successfully: {code[:50]}...")
            return result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            logger.error("Code execution timed out")
            return "", "Execution timed out."
        except Exception as e:
            logger.error(f"Error during code execution: {str(e)}")
            return "", f"Error during execution: {str(e)}"
        finally:
            os.unlink(temp_file_path)

    @error_handler
    def validate_code(self, code: str) -> bool:
        """
        验证代码是否安全。

        Args:
            code (str): 要验证的代码

        Returns:
            bool: 如果代码安全返回 True，否则返回 False
        """
        is_safe = self.security_manager.is_safe_code(code)
        if not is_safe:
            logger.warning(f"Unsafe code detected: {code[:50]}...")
        return is_safe

    @error_handler
    def get_supported_languages(self) -> List[str]:
        """
        获取支持的编程语言列表。

        Returns:
            List[str]: 支持的编程语言列表
        """
        supported_languages = ['python']
        logger.info(f"Supported languages: {supported_languages}")
        return supported_languages