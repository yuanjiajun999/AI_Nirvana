from typing import List, Dict, Tuple
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
import ast
from src.utils.error_handler import AIAssistantException, error_handler, logger
from src.utils.security import SecurityManager

class CodeExecutorInterface(ABC):
    @abstractmethod
    def execute_code(self, code: str, language: str) -> Tuple[str, str]:
        pass # pragma: no cover

    @abstractmethod
    def validate_code(self, code: str) -> bool:
        pass # pragma: no cover

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        pass # pragma: no cover

class CodeExecutor:
    def __init__(self):
        self.security_manager = SecurityManager()
        self.allowed_modules = {'math', 'random', 'datetime'}
        logger.info("CodeExecutor initialized")

    def is_safe_code(self, code: str) -> bool:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for n in node.names:
                        if n.name not in self.allowed_modules:
                            return False
                elif isinstance(node, ast.ImportFrom):
                    if node.module not in self.allowed_modules:
                        return False
                elif isinstance(node, (ast.Call, ast.Attribute)):
                    func = node.func if isinstance(node, ast.Call) else node
                    if isinstance(func, ast.Attribute):
                        if func.attr in ['open', 'exec', 'eval']:
                            return False
            return True
        except SyntaxError:
            return False

    @error_handler
    def execute_code(self, code: str, language: str) -> Tuple[str, str]:
        if language.lower() != "python":
            logger.warning(f"Unsupported language: {language}")
            return "", "Only Python execution is currently supported."

        if not self.is_safe_code(code):
            logger.warning("Unsafe code detected")
            return "", "Code execution blocked due to security concerns."

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        try:
            result = subprocess.run(
                ["python", temp_file_path], capture_output=True, text=True, timeout=5
            )
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
        is_safe = self.is_safe_code(code)
        if not is_safe:
            logger.warning(f"Unsafe code detected: {code[:50]}...")
        return is_safe

    @error_handler
    def get_supported_languages(self) -> List[str]:
        supported_languages = ["python"]
        logger.info(f"Supported languages: {supported_languages}")
        return supported_languages
class LanguageExecutor(ABC):
    @abstractmethod
    def execute(self, code: str) -> Tuple[str, str]:
        pass # pragma: no cover

class PythonExecutor(LanguageExecutor):
    @error_handler
    def execute(self, code: str) -> Tuple[str, str]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        try:
            result = subprocess.run(
                ["python", temp_file_path], capture_output=True, text=True, timeout=5
            )
            logger.info(f"Python code executed successfully: {code[:50]}...")
            return result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            logger.error("Python code execution timed out")
            return "", "Execution timed out."
        except Exception as e:
            logger.error(f"Error during Python code execution: {str(e)}")
            return "", f"Error during execution: {str(e)}"
        finally:
            os.unlink(temp_file_path)