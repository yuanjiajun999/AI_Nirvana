from src.core.multimodal import MultimodalInterface
from PIL import Image
import io

def main():
    # 初始化多模态接口
    interface = MultimodalInterface()

    # 文本处理示例
    text_result = interface.process_input("text", "This is a test sentence.")
    print("Text processing result:", text_result)

    # 语音处理示例（使用模拟的音频数据）
    audio_data = b'\x00\x01\x02\x03'  # 模拟的音频数据，使用ASCII兼容的字节
    speech_result = interface.process_input("speech", audio_data)
    print("Speech processing result:", speech_result)

    # 图像处理示例
    image = Image.new('RGB', (100, 100), color='red')
    image_result = interface.process_input("image", image)
    print("Image processing result:", image_result)

    # 批处理示例
    batch_inputs = [
        {"type": "text", "data": "First text"},
        {"type": "text", "data": "Second text"},
        {"type": "image", "data": Image.new('RGB', (50, 50), color='blue')}
    ]
    batch_results = interface.batch_process(batch_inputs)
    print("Batch processing results:", batch_results)

    # 多模态融合示例
    fusion_result = interface.multimodal_fusion("This is an image", Image.new('RGB', (200, 200), color='green'))
    print("Multimodal fusion result:", fusion_result)

if __name__ == "__main__":
    main()