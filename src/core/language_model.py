import os
from typing import List, Dict, Optional, Any
import openai
from dotenv import load_dotenv
from src.utils.error_handler import error_handler, logger, ModelError

load_dotenv()

class LanguageModel:
    def __init__(self, default_model: str = "gpt-3.5-turbo-0125"):
        self.api_key = "sk-vRu126d626325944f7040b39845200bafd41123d8f3g48Ol"  # 直接设置 API 密钥
        openai.api_key = self.api_key
        openai.api_base = "https://api.gptsapi.net/v1"
        self.default_model = default_model
        logger.info(f"LanguageModel initialized with model: {default_model}")

    @error_handler
    def generate_response(self, prompt: str, context: str = "", model: Optional[str] = None) -> str:
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
        model = model or self.default_model
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": prompt}
                ]
            )
            result = response.choices[0].message.content
            logger.info(f"Generated response for prompt: {prompt[:50]}...")
            return result
        except Exception as e:
            logger.error(f"Error in generating response: {e}")
            raise ModelError(f"Failed to generate response: {str(e)}")

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
            models = openai.Model.list()
            model_list = [model.id for model in models.data]
            logger.info(f"Retrieved available models: {model_list}")
            return model_list
        except Exception as e:
            logger.error(f"Error in fetching available models: {e}")
            raise ModelError(f"Failed to fetch available models: {str(e)}")

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
            model_info = openai.Model.retrieve(model)
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
        response = self.generate_response(prompt)
        try:
            import json
            sentiment = json.loads(response)
            logger.info(f"Sentiment analysis completed for text: {text[:50]}...")
            return sentiment
        except json.JSONDecodeError:
            logger.error("Failed to parse sentiment analysis result")
            raise ModelError("Failed to parse sentiment analysis result")

    def clear_context(self) -> None:
        """
        清除当前的对话上下文。
        """
        # 如果需要清除上下文，可以在这里实现相应的逻辑
        logger.info("Context cleared")