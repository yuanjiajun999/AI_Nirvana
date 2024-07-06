import pytest
from src.utils.security import SecurityManager

@pytest.fixture
def security_manager():
    return SecurityManager()

def test_is_safe_code(security_manager):
    safe_code = "print('Hello, World!')"
    unsafe_code = "import os; os.system('rm -rf /')"
    assert security_manager.is_safe_code(safe_code)
    assert not security_manager.is_safe_code(unsafe_code)

def test_encrypt_decrypt(security_manager):
    original_data = "sensitive information"
    encrypted = security_manager.encrypt_sensitive_data(original_data)
    decrypted = security_manager.decrypt_sensitive_data(encrypted)
    assert decrypted == original_data