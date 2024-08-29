# src/api_config.py

import os

class APIConfig:
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo-0125")
    API_KEY = os.getenv("API_KEY")
    API_BASE = os.getenv("API_BASE", "https://api.gptsapi.net/v1")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
