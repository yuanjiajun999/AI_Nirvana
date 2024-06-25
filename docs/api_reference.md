# AI Nirvana API Reference

This document provides detailed information about the AI Nirvana API.

## Table of Contents

1. [LanguageModel](#languagemodel)
2. [CodeExecutor](#codeexecutor)
3. [ContextManager](#contextmanager)
4. [SecurityManager](#securitymanager)
5. [Configuration](#configuration)

## LanguageModel

### `generate_response(prompt, context=None, max_length=150)`

Generates a response based on the given prompt and context.

- `prompt`: The input text to generate a response for.
- `context`: Optional context for the conversation.
- `max_length`: Maximum length of the generated response.

## CodeExecutor

### `execute_code(code, language)`

Executes the given code in a sandboxed environment.

- `code`: The code to execute.
- `language`: The programming language of the code.

## ContextManager

### `add_to_context(message)`

Adds a message to the conversation context.

### `get_context()`

Returns the current conversation context.

## SecurityManager

### `is_safe_code(code)`

Checks if the given code is safe to execute.

### `encrypt_sensitive_data(data)`

Encrypts sensitive data.

### `decrypt_sensitive_data(encrypted_data)`

Decrypts sensitive data.

## Configuration

AI Nirvana can be configured using the `config/default_config.yaml` file. Available options include:

- `model_name`: The name of the language model to use.
- `interface`: The interface to use (cli, gui, or api).
- `max_context_length`: Maximum number of messages to keep in context.