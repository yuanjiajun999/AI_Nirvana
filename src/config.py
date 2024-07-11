import json
import os

class Config:
    def __init__(self, config_file='config.json'):
        self.config_file = config_file
        self.config = self.load_config()
        self.predefined_responses = {
            "introduce_yourself": "Hello, I am the AI assistant created for the AI Nirvana project. I'm here to help you with a variety of tasks.",
            "how_are_you": "I'm doing well, thank you for asking.",
            "what_can_you_do": "I can assist you with a wide range of tasks, such as answering questions, generating content, and summarizing text."
        }
    
    def load_config(self):
        if not os.path.exists(self.config_file):
            return self.create_default_config()
        
        with open(self.config_file, 'r') as f:
            return json.load(f)

    def create_default_config(self):
        default_config = {
            "model": "local",
            "log_level": "INFO",
            "max_input_length": 100,
            "api_key": "",
            "api_url": ""
        }
        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=4)
        return default_config

    def get(self, key):
        return self.config.get(key)

    def set(self, key, value):
        self.config[key] = value
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)