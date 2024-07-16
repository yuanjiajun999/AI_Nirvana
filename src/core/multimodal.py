from typing import Union

import speech_recognition as sr
from PIL import Image
from torchvision.transforms import Resize


class MultimodalInterface:
    def __init__(self, text_model, speech_recognizer, vision_model):
        self.text_model = text_model
        self.speech_recognizer = speech_recognizer
        self.vision_model = vision_model

    def process_input(self, input_type: str, data: Union[str, bytes, Image.Image]):
        if input_type == "text":
            return self.text_model.generate_response(data)
        elif input_type == "speech":
            text = self.speech_recognizer.recognize_google(data)
            return self.text_model.generate_response(text)
        elif input_type == "image":
            image = Resize((224, 224))(data)
            caption = self.vision_model.generate_caption(image)
            return caption
        else:
            raise ValueError("Invalid input type. Expected 'text', 'speech', or 'image'.")