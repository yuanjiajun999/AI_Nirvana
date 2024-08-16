from typing import Union, List, Dict, Tuple
import speech_recognition as sr
from PIL import Image
import torch
from torchvision.transforms import Resize
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoFeatureExtractor, AutoModelForImageClassification
import numpy as np

class MultimodalInterface:
    def __init__(self, text_model=None, speech_recognizer=None, vision_model=None):
        self.text_model = text_model or self._load_text_model()
        self.speech_recognizer = speech_recognizer or sr.Recognizer()
        self.vision_model = vision_model or self._load_vision_model()
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

    def _load_text_model(self):
        return AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

    def _load_vision_model(self):
        return AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

    def process_input(self, input_type: Union[str, None], data: Union[str, bytes, Image.Image, List[Union[str, bytes, Image.Image, Tuple[str, Union[str, bytes, Image.Image]]]]]):
        if isinstance(data, list):
            return [self.process_single_input(item[0], item[1]) if isinstance(item, tuple) else self.process_single_input(input_type, item) for item in data]
        return self.process_single_input(input_type, data)

    def process_single_input(self, input_type: str, data: Union[str, bytes, Image.Image]):
        if input_type == "text":
            return self.process_text_input(data)
        elif input_type == "speech":
            return self.process_speech_input(data)
        elif input_type == "image":
            return self.process_image_input(data)
        else:
            raise ValueError(f"Invalid input type: {input_type}")

    def process_text_input(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.text_model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        return {"text": text, "sentiment": probabilities.tolist()[0]}

    def process_speech_input(self, audio: bytes):
        try:
            text = self.speech_recognizer.recognize_google(audio)
            sentiment = self.process_text_input(text)
            return {"speech_to_text": text, "sentiment": sentiment["sentiment"]}
        except sr.UnknownValueError:
            return {"error": "Speech not understood"}
        except sr.RequestError:
            return {"error": "Could not request results from speech recognition service"}

    def process_image_input(self, image: Image.Image):
        image = Resize((224, 224))(image)
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.vision_model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        return {"image_classification": probabilities.tolist()[0]}

    def process(self, input_data):
        if isinstance(input_data, str):
            return self.process_text(input_data)
        elif isinstance(input_data, bytes):
            return self.process_image(input_data)
        elif isinstance(input_data, Image.Image):
            return self.process_image(input_data)
        else:
            raise ValueError("Unsupported input type. Expected str, bytes, or PIL.Image.Image.")
    
    def process_single(self, input_data: Union[str, bytes, Image.Image]):
        if isinstance(input_data, str):
            return self.process_input("text", input_data)
        elif isinstance(input_data, bytes):
            return self.process_input("speech", input_data)
        elif isinstance(input_data, Image.Image):
            return self.process_input("image", input_data)
        else:
            raise ValueError("Unsupported input type. Expected str, bytes, or PIL.Image.Image.")

    def batch_process(self, input_data: List[Dict[str, Union[str, bytes, Image.Image]]]):
        results = []
        for item in input_data:
            input_type = item.get("type")
            data = item.get("data")
            if input_type and data:
                try:
                    results.append(self.process_single_input(input_type, data))
                except ValueError as e:
                    results.append({"error": str(e)})
            else:
                results.append({"error": "Invalid input format"})
        return results
    
    def multimodal_fusion(self, text: str, image: Image.Image):
        text_result = self.process_text_input(text)
        image_result = self.process_image_input(image)
        
        text_embedding = self.text_model.get_input_embeddings()(self.tokenizer(text, return_tensors="pt").input_ids)
        image_embedding = self.vision_model.get_input_embeddings()(self.feature_extractor(images=image, return_tensors="pt").pixel_values)
        
        fused_embedding = torch.cat([text_embedding.mean(dim=1), image_embedding.mean(dim=1)], dim=-1)
        
        return {
            "text_sentiment": text_result["sentiment"],
            "image_classification": image_result["image_classification"],
            "fused_embedding": fused_embedding.tolist()[0]
        }
    
    def process(self, input_data):
        try:
            if isinstance(input_data, str):
                # 直接在这里处理文本
                # 可以添加任何需要的文本处理逻辑
                return input_data  # 或者返回处理后的文本
            elif isinstance(input_data, Image.Image):
                return self.process_image(input_data)
            else:
                raise ValueError("Unsupported input type")
        except Exception as e:
            print(f"Error in MultimodalInterface.process: {str(e)}")
            raise