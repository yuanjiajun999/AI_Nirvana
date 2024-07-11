import pytest
from src.core.language_model import LanguageModel

@pytest.fixture
def language_model():
    return LanguageModel()

def test_generate_response(language_model):
    response = language_model.generate_response("Hello, how are you?")
    assert isinstance(response, str)
    assert len(response) > 0

def test_generate_response_with_context(language_model):
    context = "You are a helpful AI assistant."
    response = language_model.generate_response("What are you?", context)
    assert "assistant" in response.lower()