import os
import subprocess
import re
from cryptography.fernet import Fernet

class SecurityManager:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)

    def create_sandbox(self):
        sandbox_dir = os.path.join(os.getcwd(), 'sandbox')
        os.makedirs(sandbox_dir, exist_ok=True)
        return sandbox_dir

    def is_safe_code(self, code):
        unsafe_patterns = [
            r'import\s+os',
            r'import\s+sys',
            r'__import__\(',
            r'eval\(',
            r'exec\(',
            r'subprocess',
            r'open\('
        ]
        
        for pattern in unsafe_patterns:
            if re.search(pattern, code):
                return False
        return True

    def execute_in_sandbox(self, code, language):
        if not self.is_safe_code(code):
            return None, "Code execution blocked due to security concerns."

        sandbox_dir = self.create_sandbox()
        file_path = os.path.join(sandbox_dir, f'script.{language}')
        
        with open(file_path, 'w') as f:
            f.write(code)

        if language == 'python':
            cmd = ['python', file_path]
        elif language == 'javascript':
            cmd = ['node', file_path]
        else:
            return None, f"Unsupported language: {language}"

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            return result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return None, "Execution timed out"

    def encrypt_sensitive_data(self, data):
        return self.cipher_suite.encrypt(data.encode()).decode()

    def decrypt_sensitive_data(self, encrypted_data):
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()