import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
from unittest.mock import patch

from src.core.wildcard_api import WildCardAPI


class TestWildCardAPI(unittest.TestCase):
    def setUp(self):
        self.api_key = "your_api_key_here"
        self.api = WildCardAPI(self.api_key)

    def test_chat_completion(self):
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        response = self.api.chat_completion(model="gpt-3.5-turbo", messages=messages)
        self.assertIsNotNone(response)

    def test_embeddings(self):
        text = "The quick brown fox jumped over the lazy dog."
        embeddings = self.api.embeddings(model="text-embedding-ada-002", input=text)
        self.assertIsNotNone(embeddings)

    def test_image_generation(self):
        prompt = "A cute baby sea otter"
        images = self.api.image_generation(model="dall-e-3", prompt=prompt, n=1, size="1024x1024")
        self.assertIsNotNone(images)