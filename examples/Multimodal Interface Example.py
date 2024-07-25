from src.core.multimodal import MultimodalInterface
import numpy as np
from PIL import Image


class DummyTextModel:
    def generate_response(self, text):
        return f"Processed text: {text}"


class DummySpeechRecognizer:
    def recognize_google(self, audio):
        return "Dummy transcribed text"


class DummyVisionModel:
    def generate_caption(self, image):
        return "A dummy image caption"


def main():
    # 创建虚拟模型实例
    text_model = DummyTextModel()
    speech_recognizer = DummySpeechRecognizer()
    vision_model = DummyVisionModel()

    # 初始化多模态接口
    multimodal = MultimodalInterface(text_model, speech_recognizer, vision_model)

    # 文本处理示例
    text_input = "Hello, world!"
    text_result = multimodal.process_input("text", text_input)
    print("Text Processing Result:")
    print(text_result)

    # 语音处理示例（使用模拟的音频数据）
    audio_data = np.random.bytes(1000)
    speech_result = multimodal.process_input("speech", audio_data)
    print("\nSpeech Processing Result:")
    print(speech_result)

    # 图像处理示例（创建一个简单的测试图像）
    image = Image.new("RGB", (100, 100), color="red")
    image_result = multimodal.process_input("image", image)
    print("\nImage Processing Result:")
    print(image_result)


if __name__ == "__main__":
    main()
