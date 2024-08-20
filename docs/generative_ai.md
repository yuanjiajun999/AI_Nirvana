# GenerativeAI 类使用文档

## 概述
`GenerativeAI` 类提供了多种自然语言处理和计算机视觉功能，包括文本生成、翻译、图像分类、问答系统、情感分析和文本摘要等。

## 初始化
```python
from src.core.generative_ai import GenerativeAI

ai = GenerativeAI()
```

注意：确保在使用前已设置 `API_KEY` 和 `API_BASE` 环境变量。

## 主要方法

### 1. 文本生成
```python
generated_text = ai.generate_text(prompt, max_tokens=100, temperature=0.7)
```
- `prompt` (str): 输入提示文本
- `max_tokens` (int, 可选): 生成的最大标记数，默认为100
- `temperature` (float, 可选): 控制随机性，默认为0.7

### 2. 文本翻译
```python
translated_text = ai.translate_text(text, target_language="zh")
```
- `text` (str): 要翻译的文本
- `target_language` (str, 可选): 目标语言代码，默认为"zh"（中文）

### 3. 图像分类
```python
classification = ai.classify_image(image, top_k=5)
```
- `image` (str 或 PIL.Image.Image): 图像文件路径或PIL图像对象
- `top_k` (int, 可选): 返回的顶级预测数量，默认为5

### 4. 图像描述生成
```python
caption = ai.generate_image_caption(image)
```
- `image` (str 或 PIL.Image.Image): 图像文件路径或PIL图像对象

### 5. 问答
```python
answer = ai.answer_question(context, question)
```
- `context` (str): 上下文文本
- `question` (str): 问题

### 6. 情感分析
```python
sentiment = ai.analyze_sentiment(text)
```
- `text` (str): 要分析情感的文本

### 7. 文本摘要
```python
summary = ai.summarize_text(text, max_length=130, min_length=30)
```
- `text` (str): 要摘要的文本
- `max_length` (int, 可选): 摘要的最大长度，默认为130
- `min_length` (int, 可选): 摘要的最小长度，默认为30

### 8. 模型微调
```python
ai.fine_tune(train_texts, val_texts=None, epochs=5, learning_rate=2e-5, batch_size=4)
```
- `train_texts` (List[str]): 训练文本列表
- `val_texts` (List[str], 可选): 验证文本列表
- `epochs` (int, 可选): 训练轮数，默认为5
- `learning_rate` (float, 可选): 学习率，默认为2e-5
- `batch_size` (int, 可选): 批次大小，默认为4

### 9. 保存模型
```python
ai.save_model(path)
```
- `path` (str): 保存模型的路径

### 10. 加载模型
```python
ai.load_model(path)
```
- `path` (str): 加载模型的路径

### 11. 切换模型
```python
ai.switch_model(model_name)
```
- `model_name` (str): 要切换到的模型名称

### 12. 清理资源
```python
ai.cleanup()
```

## 注意事项
- 确保已安装所有必要的依赖，包括 `torch`, `transformers`, `PIL`, 和 `langdetect`。
- 某些功能（如翻译和图像处理）可能需要额外的模型或资源，请确保它们已正确安装和配置。
- 使用 GPU 可以显著提高性能，但请确保您的系统支持 CUDA。
- 处理错误时，类使用自定义的错误处理装饰器来捕获和记录异常。
- 在进行多次请求时，请注意 API 速率限制和令牌使用情况。

## 示例用法
请参考示例文件以了解如何在实际场景中使用 `GenerativeAI` 类的各种功能。

有关更多详细信息和高级用法，请参阅源代码中的文档字符串。