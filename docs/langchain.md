# LangChainAgent Documentation

## Overview

`LangChainAgent` is a versatile class that provides various natural language processing capabilities using the LangChain library and OpenAI's language models. It offers functionalities such as question answering, text summarization, text generation, sentiment analysis, and keyword extraction.

## Installation

Ensure you have the required dependencies installed:

```
pip install langchain openai python-dotenv
```

## Usage

To use the `LangChainAgent`, first import it from the appropriate module:

```python
from src.core.langchain import LangChainAgent
```

Then, create an instance of the `LangChainAgent`:

```python
agent = LangChainAgent()
```

## Features

### 1. Question Answering

Use the `run_qa_task` method to get answers to questions:

```python
question = "What is the capital of France?"
answer = agent.run_qa_task(question)
print(answer)
```

### 2. Text Summarization

Summarize long texts using the `run_summarization_task` method:

```python
long_text = "Your long text here..."
summary = agent.run_summarization_task(long_text)
print(summary)
```

### 3. Text Generation

Generate text based on a prompt using the `run_generation_task` method:

```python
prompt = "Write a short story about a robot learning to feel emotions."
generated_text = agent.run_generation_task(prompt)
print(generated_text)
```

### 4. Sentiment Analysis

Analyze the sentiment of a given text using the `analyze_sentiment` method:

```python
text = "I absolutely love this new restaurant! The food is amazing."
sentiment = agent.analyze_sentiment(text)
print(sentiment)
```

### 5. Keyword Extraction

Extract key keywords from a text using the `extract_keywords` method:

```python
text = "Machine learning is a subset of artificial intelligence..."
keywords = agent.extract_keywords(text)
print(keywords)
```

## Error Handling

All methods in `LangChainAgent` include error handling. If an API error occurs, the methods will return an appropriate error message instead of raising an exception.

## Configuration

The `LangChainAgent` uses environment variables for configuration. Ensure you have a `.env` file in your project root with the following variables:

```
API_KEY=your_openai_api_key
API_BASE=https://api.openai.com/v1
MODEL_NAME=gpt-3.5-turbo
TEMPERATURE=0.7
MAX_TOKENS=256
```

Adjust these values as needed for your specific use case.

## Best Practices

1. Keep your API key secure and never share it publicly.
2. Be mindful of the API usage limits and costs associated with OpenAI's services.
3. For production use, implement proper error handling and logging.
4. Consider implementing rate limiting to avoid exceeding API quotas.

## Limitations

- The quality of results depends on the underlying language model and the prompts used.
- The agent's knowledge is limited to the training data of the language model.
- API calls can be expensive for large-scale applications.

## Support

For any issues or feature requests, please open an issue in the project's GitHub repository.