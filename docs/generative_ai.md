GenerativeAI 使用文档
GenerativeAI 类提供了多种自然语言处理和计算机视觉功能。以下是主要功能及其使用方法：
初始化
from src.core.generative_ai import GenerativeAI

ai = GenerativeAI(model_name="gpt2", device="cuda")

model_name: 默认为 "gpt2"
device: 默认为 "cuda" （如果可用），否则为 "cpu"

文本生成
generated_text = ai.generate_text("Your prompt here", max_length=100, num_return_sequences=1)
文本翻译
translated_text = ai.translate_text("Hello, world!", target_language="zh")
图像分类
classification = ai.classify_image("path/to/image.jpg", top_k=5)
问答
answer = ai.answer_question("Your context here", "Your question here")
情感分析
sentiment = ai.analyze_sentiment("Your text here")
文本摘要
summary = ai.summarize_text("Your long text here", max_length=130, min_length=30)
注意事项

确保已安装所有必要的依赖，包括 torch, transformers, 和 PIL。
某些功能（如翻译和图像处理）可能需要额外的模型或资源，请确保它们已正确安装和配置。
使用 GPU 可以显著提高性能，但请确保您的系统支持 CUDA。

有关更多详细信息和高级用法，请参阅源代码文档字符串。