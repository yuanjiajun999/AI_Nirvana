import unittest
from unittest.mock import patch, MagicMock
import torch
from PIL import Image
import pytest
from src.core.generative_ai import GenerativeAI, CustomDataset

class TestGenerativeAI(unittest.TestCase):
    @classmethod
    @patch('src.core.generative_ai.AutoModelForCausalLM.from_pretrained')
    @patch('src.core.generative_ai.AutoTokenizer.from_pretrained')
    @patch('src.core.generative_ai.pipeline')
    def setUpClass(cls, mock_pipeline, mock_tokenizer, mock_model):
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        mock_pipeline.side_effect = [
            MagicMock(),  # text_generation_pipeline
            ValueError("Failed to load translation pipeline"),
            ValueError("Failed to load image classification pipeline"),
            ValueError("Failed to load image captioning pipeline")
        ]
        cls.ai = GenerativeAI(model_name="gpt2")
        # Manually set up pipelines for testing
        cls.ai.text_generation_pipeline = MagicMock()
        cls.ai.image_classification_pipeline = MagicMock()
        cls.ai.image_captioning_pipeline = MagicMock()

    def setUp(self):
        # Reset mock for each test
        self.ai.text_generation_pipeline.reset_mock()
        self.ai.text_generation_pipeline.side_effect = None

    def test_init(self):
        self.assertIsNotNone(self.ai.model)
        self.assertIsNotNone(self.ai.tokenizer)
        self.assertEqual(self.ai.model_name, "gpt2")
        self.assertIsNotNone(self.ai.text_generation_pipeline)
        self.assertIsNone(self.ai.translation_pipeline)
        self.assertIsNotNone(self.ai.image_classification_pipeline)
        self.assertIsNotNone(self.ai.image_captioning_pipeline)

    def test_generate_text(self):
        mock_result = [{'generated_text': 'Test generated text'}]
        self.ai.text_generation_pipeline.return_value = mock_result
        result = self.ai.generate_text("Test prompt")
        self.assertEqual(result, ['Test generated text'])
        self.ai.text_generation_pipeline.assert_called_once()

    def test_translate_text(self):
        text = "Hello, world!"
        result = self.ai.translate_text(text)
        self.assertEqual(result, text, "When translation pipeline is not available, original text should be returned")

    def test_classify_image(self):
        mock_image = MagicMock(spec=Image.Image)
        mock_result = [{'label': 'cat', 'score': 0.9}]
        self.ai.image_classification_pipeline.return_value = mock_result
        result = self.ai.classify_image(mock_image)
        self.assertEqual(result, mock_result)
        self.ai.image_classification_pipeline.assert_called_once_with(mock_image, top_k=5)

    @patch('src.core.generative_ai.train_test_split')
    @patch('src.core.generative_ai.DataLoader')
    @patch('src.core.generative_ai.AdamW')
    @patch('src.core.generative_ai.tqdm')
    def test_fine_tune(self, mock_tqdm, mock_adamw, mock_dataloader, mock_train_test_split):
        mock_train_test_split.return_value = (["train1", "train2"], ["val1"])
        mock_dataloader.return_value = [{"input_ids": torch.tensor([1, 2, 3]), "attention_mask": torch.tensor([1, 1, 1])}]
        mock_adamw.return_value = MagicMock()
        mock_tqdm.return_value = [{"input_ids": torch.tensor([1, 2, 3]), "attention_mask": torch.tensor([1, 1, 1])}]

        self.ai.model = MagicMock()
        self.ai.model.return_value = MagicMock(loss=torch.tensor(0.5, requires_grad=True))

        train_data = ["Example text 1", "Example text 2", "Example text 3"]
        self.ai.fine_tune(train_data, epochs=1)

        mock_train_test_split.assert_called_once()
        mock_dataloader.assert_called()
        mock_adamw.assert_called_once()
        self.ai.model.assert_called()

    @patch('src.core.generative_ai.AutoModelForCausalLM')
    @patch('src.core.generative_ai.AutoTokenizer')
    def test_save_and_load_model(self, mock_tokenizer, mock_model):
        mock_model_instance = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        self.ai.model = mock_model_instance
        self.ai.tokenizer = mock_tokenizer_instance

        self.ai.save_model("dummy_path")
        mock_model_instance.save_pretrained.assert_called_once_with("dummy_path")
        mock_tokenizer_instance.save_pretrained.assert_called_once_with("dummy_path")

        self.ai.load_model("dummy_path")
        mock_model.from_pretrained.assert_called_with("dummy_path")
        mock_tokenizer.from_pretrained.assert_called_with("dummy_path")

    @patch('PIL.Image.open')
    def test_generate_image_caption(self, mock_image_open):
        mock_image = MagicMock(spec=Image.Image)
        mock_image_open.return_value = mock_image
        mock_result = [{'generated_text': 'A cat sitting on a couch'}]
        self.ai.image_captioning_pipeline.return_value = mock_result
        
        # Test with file path
        result = self.ai.generate_image_caption("dummy_image.jpg")
        self.assertEqual(result, 'A cat sitting on a couch')
        
        # Test with Image object
        result = self.ai.generate_image_caption(mock_image)
        self.assertEqual(result, 'A cat sitting on a couch')
        
        self.ai.image_captioning_pipeline.assert_called_with(mock_image)

    @patch('src.core.generative_ai.pipeline')
    def test_answer_question(self, mock_pipeline):
        mock_qa_pipeline = MagicMock(return_value={'answer': 'Paris'})
        mock_pipeline.return_value = mock_qa_pipeline
        result = self.ai.answer_question("The capital of France is Paris.", "What is the capital of France?")
        self.assertEqual(result, 'Paris')
        mock_qa_pipeline.assert_called_once_with(question="What is the capital of France?", context="The capital of France is Paris.")

    @patch('src.core.generative_ai.pipeline')
    def test_analyze_sentiment(self, mock_pipeline):
        mock_sentiment_pipeline = MagicMock(return_value=[{'label': 'POSITIVE', 'score': 0.9}])
        mock_pipeline.return_value = mock_sentiment_pipeline
        result = self.ai.analyze_sentiment("I love this product!")
        self.assertEqual(result, {'label': 'POSITIVE', 'score': 0.9})
        mock_sentiment_pipeline.assert_called_once_with("I love this product!")

    @patch('src.core.generative_ai.pipeline')
    def test_summarize_text(self, mock_pipeline):
        mock_summarizer = MagicMock(return_value=[{'summary_text': 'This is a summary.'}])
        mock_pipeline.return_value = mock_summarizer
        result = self.ai.summarize_text("This is a long text that needs to be summarized.")
        self.assertEqual(result, 'This is a summary.')
        mock_summarizer.assert_called_once_with("This is a long text that needs to be summarized.", max_length=130, min_length=30, do_sample=False)

    def test_custom_dataset(self):
        texts = ["Text 1", "Text 2", "Text 3"]
        tokenizer = MagicMock()
        tokenizer.return_value = {'input_ids': torch.tensor([1, 2, 3]), 'attention_mask': torch.tensor([1, 1, 1])}
        dataset = CustomDataset(texts, tokenizer, max_length=10)
        
        self.assertEqual(len(dataset), 3)
        item = dataset[0]
        self.assertIn('input_ids', item)
        self.assertIn('attention_mask', item)
        tokenizer.assert_called_with("Text 1", truncation=True, padding='max_length', max_length=10, return_tensors='pt')

    @patch('src.core.generative_ai.logger')
    def test_logging(self, mock_logger):
        mock_result = [{'generated_text': 'Test generated text'}]
        self.ai.text_generation_pipeline.return_value = mock_result
        self.ai.generate_text("Test prompt")
        mock_logger.info.assert_called()

    def test_error_handling(self):
        self.ai.text_generation_pipeline.side_effect = Exception("Test error")
        with self.assertRaises(Exception):
            self.ai.generate_text("Test prompt")

if __name__ == '__main__':
    unittest.main()

if __name__ == '__main__':
    unittest.main()
    
    @patch('PIL.Image.open')
    def test_generate_image_caption(self, mock_image_open):
        mock_image = MagicMock(spec=Image.Image)
        mock_image_open.return_value = mock_image
        self.ai.image_captioning_pipeline = MagicMock(return_value=[{'generated_text': 'A cat sitting on a couch'}])
        
        # Test with file path
        result = self.ai.generate_image_caption("dummy_image.jpg")
        assert result == 'A cat sitting on a couch'
        
        # Test with Image object
        result = self.ai.generate_image_caption(mock_image)
        assert result == 'A cat sitting on a couch'

    @patch('src.core.generative_ai.pipeline')
    def test_answer_question(self, mock_pipeline):
        mock_qa = MagicMock(return_value={'answer': 'Paris'})
        mock_pipeline.return_value = mock_qa
        
        result = self.ai.answer_question("The capital of France is Paris.", "What is the capital of France?")
        assert result == 'Paris'
        mock_qa.assert_called_once()

    @patch('src.core.generative_ai.pipeline')
    def test_analyze_sentiment(self, mock_pipeline):
        mock_sentiment = MagicMock(return_value=[{'label': 'POSITIVE', 'score': 0.9}])
        mock_pipeline.return_value = mock_sentiment
        
        result = self.ai.analyze_sentiment("I love this product!")
        assert result == {'label': 'POSITIVE', 'score': 0.9}
        mock_sentiment.assert_called_once()

    @patch('src.core.generative_ai.pipeline')
    def test_summarize_text(self, mock_pipeline):
        mock_summarizer = MagicMock(return_value=[{'summary_text': 'This is a summary.'}])
        mock_pipeline.return_value = mock_summarizer
        
        result = self.ai.summarize_text("This is a long text that needs to be summarized.")
        assert result == 'This is a summary.'
        mock_summarizer.assert_called_once()