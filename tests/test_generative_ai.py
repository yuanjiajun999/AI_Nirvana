import unittest
from unittest.mock import patch, MagicMock
from src.core.generative_ai import GenerativeAI
import torch

class TestGenerativeAI(unittest.TestCase):

    @patch('src.core.generative_ai.OpenAI')
    def setUp(self, MockOpenAI):
        # 创建GenerativeAI实例
        self.ai = GenerativeAI()

    def test_generate_text(self):
        prompt = "The quick brown fox"
        expected_output = "The quick brown fox jumps over the lazy dog."

        # 模拟OpenAI的返回值
        self.ai.client.chat.completions.create = MagicMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content=expected_output))],
            usage=MagicMock(total_tokens=10)
        ))

        result = self.ai.generate_text(prompt)
        self.assertEqual(result, expected_output)
        self.ai.client.chat.completions.create.assert_called_once()

    def test_translate_text(self):
        text = "Hello, world!"
        target_language = "zh"
        expected_translation = "你好，世界！"

        # 模拟翻译管道的返回值
        self.ai.translation_pipeline = MagicMock(return_value=[{"translation_text": expected_translation}])

        translation = self.ai.translate_text(text, target_language)
        self.assertEqual(translation, expected_translation)
        self.ai.translation_pipeline.assert_called_once()

    @patch('PIL.Image.open')
    def test_classify_image(self, mock_image_open):
        image_path = "test_image.jpg"
        expected_classification = [{"label": "gown", "score": 0.1953}]

        # 模拟图像分类管道的返回值
        self.ai.image_classification_pipeline = MagicMock(return_value=expected_classification)

        classification = self.ai.classify_image(image_path)
        self.assertEqual(classification, expected_classification)
        self.ai.image_classification_pipeline.assert_called_once()
        mock_image_open.assert_called_once_with(image_path)

    def test_fine_tune(self):
        train_data = ["The sky is blue.", "The sun is bright."]
    
        # 模拟 finetune_model 和 tokenizer
        mock_model = MagicMock()
        mock_parameters = [torch.tensor([1.0, 2.0], requires_grad=True)]
        mock_model.parameters.return_value = mock_parameters
        mock_tokenizer = MagicMock()
    
        # 配置 mock 的返回值
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[0, 1, 2], [3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 1]])
        }
    
        self.ai.finetune_model = mock_model
        self.ai.tokenizer = mock_tokenizer

        # 调用 fine_tune 方法
        self.ai.fine_tune(train_data, epochs=1, learning_rate=2e-5, batch_size=2)
    
        # 验证 finetune_model 的 train 方法是否被调用
        mock_model.train.assert_called_once()

    @patch('PIL.Image.open')
    def test_generate_image_caption(self, mock_image_open):
        image_path = "test_image.jpg"
        expected_caption = "a woman in a blue dress and a gold headpiece"

        # 模拟图像描述生成管道的返回值
        self.ai.image_captioning_pipeline = MagicMock(return_value=[{"generated_text": expected_caption}])

        caption = self.ai.generate_image_caption(image_path)
        self.assertEqual(caption, expected_caption)
        self.ai.image_captioning_pipeline.assert_called_once()
        mock_image_open.assert_called_once_with(image_path)

    def test_answer_question(self):
        context = "The capital of France is Paris."
        question = "What is the capital of France?"
        expected_answer = "Paris"

        # 模拟问答管道的返回值
        qa_pipeline = MagicMock(return_value={"answer": expected_answer})
        with patch('src.core.generative_ai.pipeline', return_value=qa_pipeline):
            answer = self.ai.answer_question(context, question)
            self.assertEqual(answer, expected_answer)
            qa_pipeline.assert_called_once_with(question=question, context=context)

    def test_analyze_sentiment(self):
        text = "I love this product!"
        expected_sentiment = {"label": "POSITIVE", "score": 0.999}

        # 模拟情感分析管道的返回值
        sentiment_pipeline = MagicMock(return_value=[expected_sentiment])
        with patch('src.core.generative_ai.pipeline', return_value=sentiment_pipeline):
            sentiment = self.ai.analyze_sentiment(text)
            self.assertEqual(sentiment, expected_sentiment)
            sentiment_pipeline.assert_called_once_with(text)

    def test_summarize_text(self):
        text = "Long text to be summarized..." * 10
        expected_summary = "This is a summary."

        # 模拟文本摘要管道的返回值
        summarizer = MagicMock(return_value=[{"summary_text": expected_summary}])
        with patch('src.core.generative_ai.pipeline', return_value=summarizer):
            summary = self.ai.summarize_text(text)
            self.assertEqual(summary, expected_summary)
            summarizer.assert_called_once_with(text, max_length=130, min_length=30, do_sample=False)

    @patch('src.core.generative_ai.AutoModelForCausalLM.from_pretrained')
    @patch('src.core.generative_ai.AutoTokenizer.from_pretrained')
    def test_load_model(self, mock_tokenizer, mock_model):
        path = "test_model_path"
        self.ai.load_model(path)
        mock_model.assert_called_once_with(path)
        mock_tokenizer.assert_called_once_with(path)
        self.assertIsNotNone(self.ai.finetune_model)
        self.assertIsNotNone(self.ai.tokenizer)

    @patch('src.core.generative_ai.AutoModelForCausalLM.from_pretrained')  
    @patch('src.core.generative_ai.AutoTokenizer.from_pretrained')  
    def test_switch_model(self, mock_tokenizer, mock_model):    
        # 测试 API 模型  
        api_model_name = "gpt2"  
        self.ai.switch_model(api_model_name)  
        mock_model.assert_not_called()  
        mock_tokenizer.assert_not_called()  

        # 测试本地模型  
        local_model_name = "custom_model"  
        self.ai.switch_model(local_model_name)  
        mock_model.assert_called_once()  
        mock_tokenizer.assert_called_once()

    @patch('torch.cuda.empty_cache')
    def test_cleanup(self, mock_empty_cache):
        # 确保所有属性都存在
        self.ai.model = MagicMock()
        self.ai.tokenizer = MagicMock()
        self.ai.translation_pipeline = MagicMock()
        self.ai.image_classification_pipeline = MagicMock()
        self.ai.image_captioning_pipeline = MagicMock()

        self.ai.cleanup()
        mock_empty_cache.assert_called_once()

if __name__ == '__main__':
    unittest.main()
