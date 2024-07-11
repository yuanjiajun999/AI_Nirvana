import re
from cryptography.fernet import Fernet

class SecurityManager:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)

    def is_safe_code(self, code):
        unsafe_patterns = [
            r'import\s+os',
            r'import\s+subprocess',
            r'open\(',
            r'eval\(',
            r'exec\('
        ]
        return not any(re.search(pattern, code) for pattern in unsafe_patterns)

    def encrypt_sensitive_data(self, data):
        return self.cipher_suite.encrypt(data.encode()).decode()

    def decrypt_sensitive_data(self, encrypted_data):
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()