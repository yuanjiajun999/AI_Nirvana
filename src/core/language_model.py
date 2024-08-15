import os
from typing import Any, Dict, List, Optional
import json
from src.core.model_interface import ModelInterface
from dotenv import load_dotenv
from openai import OpenAI

from src.utils.error_handler import ModelError, error_handler, logger

load_dotenv()

class LanguageModel:
    def __init__(self, model_name: str = "gpt-3.5-turbo-0125"):
        self.model_name = model_name
        self.api_key = os.getenv("API_KEY")
        self.client = OpenAI(api_key=self.api_key, base_url=os.getenv("API_BASE"))
        logger.info(f"LanguageModel initialized with model: {self.model_name}")

    @error_handler
    def generate_response(self, prompt: str, model: Optional[str] = None, context: Optional[str] = None) -> str:
        """
        生成响应。

        Args:
            prompt (str): 用户输入的提示
            context (str, optional): 对话上下文
            model (str, optional): 使用的模型名称

        Returns:
            str: 生成的响应

        Raises:
            ModelError: 如果生成响应时发生错误
        """
        model = model or self.model_name
        messages = [{"role": "system", "content": context or "You are a helpful assistant."},
                    {"role": "user", "content": prompt}]
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            raise ModelError(f"Error generating response: {str(e)}")

        
    @error_handler
    def get_available_models(self) -> List[str]:
        """
        获取可用的语言模型列表。

        Returns:
            List[str]: 可用模型的列表

        Raises:
            ModelError: 如果获取模型列表失败
        """
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Error retrieving available models: {str(e)}")
            return []

    @error_handler
    def change_model(self, new_model: str) -> None:
        if new_model in self.get_available_models():
            self.model_name = new_model
            logger.info(f"Model changed to: {self.model_name}")
        else:
            raise ValueError(f"Model {new_model} is not available.")

    @error_handler
    def change_default_model(self, model: str) -> None:
        """
        更改默认使用的语言模型。

        Args:
            model (str): 新的默认模型名称

        Raises:
            ModelError: 如果更改模型失败
        """
        if model in self.get_available_models():
            self.default_model = model
            logger.info(f"Default model changed to: {model}")
        else:
            logger.error(f"Model {model} is not available")
            raise ModelError(f"Model {model} is not available")

    @error_handler
    def get_model_info(self, model: Optional[str] = None) -> Dict[str, Any]:
        """
        获取模型信息。

        Args:
            model (str, optional): 模型名称，如果不提供则使用默认模型

        Returns:
            Dict[str, Any]: 包含模型信息的字典

        Raises:
            ModelError: 如果获取模型信息失败
        """
        model = model or self.default_model
        try:
            model_info = self.client.models.retrieve(model)
            logger.info(f"Retrieved info for model: {model}")
            return {
                "id": model_info.id,
                "created": model_info.created,
                "owned_by": model_info.owned_by,
            }
        except Exception as e:
            logger.error(f"Error in fetching model info: {e}")
            raise ModelError(f"Failed to fetch model info: {str(e)}")

    @error_handler
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        分析文本的情感。

        Args:
            text (str): 需要分析情感的文本

        Returns:
            Dict[str, float]: 包含情感分析结果的字典

        Raises:
            ModelError: 如果情感分析失败
        """
        prompt = f"Analyze the sentiment of the following text and return a JSON object with keys 'positive', 'neutral', and 'negative', where the values are floats representing the probability of each sentiment:\n\n{text}"
        try:
            response = self.generate_response(prompt)
            sentiment = json.loads(response)
            logger.info(f"Sentiment analysis completed for text: {text[:50]}...")
            return sentiment
        except json.JSONDecodeError:
            logger.error("Failed to parse sentiment analysis result")
            raise ModelError("Failed to parse sentiment analysis result")
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            raise ModelError(f"Failed to analyze sentiment: {str(e)}")

    @error_handler
    def summarize(self, text: str, max_length: int = 100) -> str:
        """
        生成文本摘要。

        Args:
            text (str): 需要摘要的文本
            max_length (int): 摘要的最大长度

        Returns:
            str: 生成的摘要

        Raises:
            ModelError: 如果生成摘要失败
        """
        prompt = f"请用中文简洁地总结以下文本，不超过{max_length}字：\n\n{text}"
        try:
            summary = self.generate_response(prompt)
            logger.info(f"Summary generated for text: {text[:50]}...")
            return summary
        except Exception as e:
            logger.error(f"Error in generating summary: {e}")
            raise ModelError(f"Failed to generate summary: {str(e)}")

    @error_handler
    def translate(self, text: str, target_language: str) -> str:
        """
        翻译文本到目标语言。

        Args:
            text (str): 需要翻译的文本
            target_language (str): 目标语言

        Returns:
            str: 翻译后的文本

        Raises:
            ModelError: 如果翻译失败
        """
        prompt = f"Translate the following text to {target_language}:\n\n{text}"
        try:
            translated_text = self.generate_response(prompt)
            logger.info(f"Text translated to {target_language}")
            return translated_text
        except Exception as e:
            logger.error(f"Error in translation: {e}")
            raise ModelError(f"Failed to translate text: {str(e)}")

    def clear_context(self) -> None:
        """
        清除当前的对话上下文。
        """
        # 如果需要清除上下文，可以在这里实现相应的逻辑
        logger.info("Context cleared")