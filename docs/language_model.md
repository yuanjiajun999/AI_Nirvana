# Language Model Module

The `LanguageModel` class provides a wrapper for interacting with advanced language models, primarily using the OpenAI API. It offers various natural language processing capabilities including text generation, sentiment analysis, summarization, and translation.

## Class: LanguageModel

### Initialization

```python
lm = LanguageModel(default_model="gpt-3.5-turbo-0125")

default_model: The name of the default model to use (default is "gpt-3.5-turbo-0125")

Methods
generate_response(prompt: str, context: str = "", model: Optional[str] = None) -> str
Generates a response based on the given prompt and optional context.
get_available_models() -> List[str]
Returns a list of available language models.
change_default_model(model: str) -> None
Changes the default model used for generating responses.
get_model_info(model: Optional[str] = None) -> Dict[str, Any]
Retrieves information about the specified model or the default model if not specified.
analyze_sentiment(text: str) -> Dict[str, float]
Analyzes the sentiment of the given text, returning a dictionary with sentiment scores.
summarize(text: str, max_length: int = 100) -> str
Generates a summary of the given text, with a specified maximum length.
translate(text: str, target_language: str) -> str
Translates the given text to the specified target language.
clear_context() -> None
Clears the current conversation context (if implemented).
Error Handling
All methods use the @error_handler decorator to catch and log errors, raising a ModelError in case of failures.
Usage Example
pythonCopyfrom src.core.language_model import LanguageModel

lm = LanguageModel()

# Generate a response
response = lm.generate_response("What is the capital of France?")
print(response)

# Analyze sentiment
sentiment = lm.analyze_sentiment("I love this product!")
print(sentiment)

# Generate a summary
summary = lm.summarize("Long text to be summarized...", max_length=50)
print(summary)

# Translate text
translated = lm.translate("Hello, world!", "French")
print(translated)
Note: Ensure that you have set up the necessary API credentials and environment variables before using this module.