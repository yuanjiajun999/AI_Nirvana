class SecurityManager:
    def __init__(self):
        pass

    def is_safe(self, text):
        return True

    def is_safe_code(self, code):
        return True

    def encrypt_sensitive_data(self, data):
        return data

    def decrypt_sensitive_data(self, data):
        return data

    def execute_in_sandbox(self, code, language):
        return "Executed in sandbox", None