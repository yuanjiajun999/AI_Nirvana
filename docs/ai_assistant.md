# AI Assistant Module

The `AIAssistant` class is a comprehensive AI assistant that provides multi-modal input processing, response generation, text summarization, sentiment analysis, and other advanced features. It also supports context management, security checks, knowledge retrieval, and reasoning.

## Class: AIAssistant

### Initialization

```python
assistant = AIAssistant(model_name="gpt-3.5-turbo", max_context_length=5)

model_name: The name of the language model to use (default is "gpt-3.5-turbo")
max_context_length: The maximum number of conversation turns to keep in context (default is 5)

Main Methods
process_input(input_data: Any) -> str
Processes multi-modal input data and generates a response.
generate_response(prompt: str) -> str
Generates a response based on the given prompt and conversation context.
summarize(text: str) -> str
Generates a summary of the given text.
analyze_sentiment(text: str) -> Dict[str, float]
Analyzes the sentiment of the given text, returning a dictionary with sentiment scores.
clear_context() -> None
Clears the current conversation context.
change_model(model_name: str) -> None
Changes the language model used for generating responses.
get_available_models() -> List[str]
Returns a list of available language models.
Security Methods
encrypt_sensitive_data(data: str) -> str
Encrypts sensitive data.
decrypt_sensitive_data(encrypted_data: str) -> str
Decrypts previously encrypted data.
execute_code(code: str, language: str) -> tuple
Safely executes code in a sandbox environment.
Advanced Features
plan_task(task_description: str) -> str
Generates a step-by-step plan for a given task description.
reinforcement_learning_action(state: Any) -> Any
Selects an action using reinforcement learning based on the current state.
active_learning_sample(unlabeled_data: List[Any]) -> Any
Selects a sample for labeling using active learning techniques.
Error Handling
All methods use the @error_handler decorator to catch and log errors, raising appropriate exceptions such as InputValidationError, ModelError, or AIAssistantException.
Usage Example
from src.core.ai_assistant import AIAssistant

assistant = AIAssistant()

# Process multi-modal input
response = assistant.process_input("What's the weather like today?")
print(response)

# Generate a response
response = assistant.generate_response("Tell me a joke")
print(response)

# Summarize text
summary = assistant.summarize("Long text to be summarized...")
print(summary)

# Analyze sentiment
sentiment = assistant.analyze_sentiment("I love this product!")
print(sentiment)

# Change the language model
assistant.change_model("gpt-4")

# Execute code safely
result, error = assistant.execute_code("print('Hello, World!')", "python")
print(result)

# Generate a task plan
plan = assistant.plan_task("Organize a team building event")
print(plan)
Note: Ensure that you have set up the necessary dependencies, API credentials, and environment variables before using this module.