import pytest
from src.core.code_executor import CodeExecutor

@pytest.fixture
def code_executor():
    return CodeExecutor()

def test_execute_python_code(code_executor):
    code = "print('Hello, World!')"
    stdout, stderr = code_executor.execute_code(code, 'python')
    assert stdout.strip() == "Hello, World!"
    assert stderr == ""

def test_execute_unsafe_code(code_executor):
    unsafe_code = "import os; os.system('rm -rf /')"
    stdout, stderr = code_executor.execute_code(unsafe_code, 'python')
    assert "security concerns" in stderr