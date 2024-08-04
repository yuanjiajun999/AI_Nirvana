import unittest
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image
import torch
import sys
import os
import speech_recognition as sr

# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.multimodal import MultimodalInterface


class TestMultimodalInterface(unittest.TestCase):
    def setUp(self):
        self.interface = MultimodalInterface()

    def test_process_text_input(self):
        result = self.interface.process_text_input("This is a test sentence.")
        self.assertIn("text", result)
        self.assertIn("sentiment", result)
        self.assertEqual(len(result["sentiment"]), 2)  # Assuming binary classification

    @patch('speech_recognition.Recognizer.recognize_google')
    def test_process_speech_input(self, mock_recognize):
        mock_recognize.return_value = "This is a test sentence."
        result = self.interface.process_speech_input(b"dummy audio data")
        self.assertIn("speech_to_text", result)
        self.assertIn("sentiment", result)

    def test_process_image_input(self):
        image = Image.new('RGB', (100, 100))
        result = self.interface.process_image_input(image)
        self.assertIn("image_classification", result)
        self.assertEqual(len(result["image_classification"]), 1000)  # Assuming ImageNet classification

    def test_process_input_text(self):
        result = self.interface.process_input("text", "This is a test sentence.")
        self.assertIn("text", result)
        self.assertIn("sentiment", result)

    @patch('speech_recognition.Recognizer.recognize_google')
    def test_process_input_speech(self, mock_recognize):
        mock_recognize.return_value = "This is a test sentence."
        result = self.interface.process_input("speech", b"dummy audio data")
        self.assertIn("speech_to_text", result)
        self.assertIn("sentiment", result)

    def test_process_input_image(self):
        image = Image.new('RGB', (100, 100))
        result = self.interface.process_input("image", image)
        self.assertIn("image_classification", result)

    def test_process_input_invalid(self):
        with self.assertRaises(ValueError):
            self.interface.process_input("invalid", "data")

    def test_process_single(self):
        result = self.interface.process_single("This is a test sentence.")
        self.assertIn("text", result)
        self.assertIn("sentiment", result)

    def test_process_list(self):
        inputs = ["Text 1", "Text 2"]
        results = self.interface.process(inputs)
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIn("text", result)
            self.assertIn("sentiment", result)

    def test_batch_process(self):
        inputs = [
            {"type": "text", "data": "Text 1"},
            {"type": "text", "data": "Text 2"}
        ]
        results = self.interface.batch_process(inputs)
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIn("text", result)
            self.assertIn("sentiment", result)

    def test_multimodal_fusion(self):
        text = "This is a test sentence."
        image = Image.new('RGB', (100, 100))
        result = self.interface.multimodal_fusion(text, image)
        self.assertIn("text_sentiment", result)
        self.assertIn("image_classification", result)
        self.assertIn("fused_embedding", result)

    def test_load_text_model(self):
        model = self.interface._load_text_model()
        self.assertIsNotNone(model)

    def test_load_vision_model(self):
        model = self.interface._load_vision_model()
        self.assertIsNotNone(model)
    
    def test_process_input_list(self):
        inputs = [
            ("text", "This is a test sentence."),
            ("image", Image.new('RGB', (100, 100)))
        ]
        results = self.interface.process_input(None, inputs)
        self.assertEqual(len(results), 2)
        self.assertIn("text", results[0])
        self.assertIn("image_classification", results[1])
        
    @patch('speech_recognition.Recognizer.recognize_google')
    def test_process_speech_input_error(self, mock_recognize):
        mock_recognize.side_effect = sr.UnknownValueError()
        result = self.interface.process_speech_input(b"dummy audio data")
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Speech not understood")

        mock_recognize.side_effect = sr.RequestError()
        result = self.interface.process_speech_input(b"dummy audio data")
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Could not request results from speech recognition service")   

    def test_batch_process_error(self):
        inputs = [
            {"type": "invalid", "data": "Invalid data"},
            {"type": "text", "data": "Valid text"}
        ]
        results = self.interface.batch_process(inputs)
        self.assertEqual(len(results), 2)
        self.assertIn("error", results[0])
        self.assertIn("Invalid input type", results[0]["error"])
        self.assertIn("text", results[1])   

    def test_process_single_input(self):
        image = Image.new('RGB', (100, 100))
        result = self.interface.process(image)
        self.assertIn("image_classification", result)  

    def test_process_single_invalid_input(self):
        with self.assertRaises(ValueError):
            self.interface.process_single(123)  # 整数不是有效的输入类型       

if __name__ == '__main__':
    unittest.main()
