import subprocess
import os
from ai_nirvana.utils.security import SecurityManager

class CodeExecutor:
    def __init__(self):
        self.security_manager = SecurityManager()

    def execute_code(self, code, language):
        return self.security_manager.execute_in_sandbox(code, language)