# examples/multimodal_example.py

from src.core.multimodal import MultimodalInterface
import numpy as np
from PIL import Image

def main():
    multimodal = MultimodalInterface()

    # 文本处理示例
    text_input = "The quick brown fox jumps over the lazy dog."
    text_result = multimodal.process_input("text", text_input)
    print("Text Processing Result:")
    print(text_result)
    print()

    # 语音处理示例（这里我们用模拟的音频数据）
    audio_data = np.random.bytes(1000)
    speech_result = multimodal.process_input("speech", audio_data)
    print("Speech Processing Result:")
    print(speech_result)
    print()

    # 图像处理示例（这里我们创建一个简单的测试图像）
    image = Image.new('RGB', (100, 100), color = 'red')
    image_result = multimodal.process_input("image", image)
    print("Image Processing Result:")
    print(image_result)

if __name__ == "__main__":
    main()