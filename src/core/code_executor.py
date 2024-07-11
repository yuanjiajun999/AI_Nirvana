import subprocess
import tempfile
import os

class CodeExecutor:
    def execute_code(self, code, language):
        if language.lower() != 'python':
            return "", "Only Python execution is currently supported."

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        try:
            result = subprocess.run(['python', temp_file_path], 
                                    capture_output=True, 
                                    text=True, 
                                    timeout=5)
            return result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return "", "Execution timed out."
        finally:
            os.unlink(temp_file_path)