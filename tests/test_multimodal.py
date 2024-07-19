import unittest
from unittest.mock import Mock
from src.core.multimodal import MultimodalInterface
from PIL import Image

class TestMultimodalInterface(unittest.TestCase):
    def setUp(self):
        self.text_model = Mock()
        self.speech_recognizer = Mock()
        self.vision_model = Mock()
        self.interface = MultimodalInterface(self.text_model, self.speech_recognizer, self.vision_model)

    def test_process_text_input(self):
        self.text_model.generate_response.return_value = "Text response"
        result = self.interface.process_input("text", "Hello, world!")
        self.assertEqual(result, "Text response")

    def test_process_speech_input(self):
        self.speech_recognizer.recognize_google.return_value = "Recognized speech"
        self.text_model.generate_response.return_value = "Speech response"
        result = self.interface.process_input("speech", b"audio_data")
        self.assertEqual(result, "Speech response")

    def test_process_image_input(self):
        self.vision_model.generate_caption.return_value = "Image caption"
        image = Image.new('RGB', (100, 100))
        result = self.interface.process_input("image", image)
        self.assertEqual(result, "Image caption")

if __name__ == '__main__':
    unittest.main()