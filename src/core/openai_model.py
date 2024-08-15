from .model_interface import ModelInterface
from src.core.openai_api import OpenAIAPI

class OpenAIModel(ModelInterface):
    def generate_response(self, prompt: str, context: dict = None) -> str:
        return OpenAIAPI.generate_response(prompt, context)

    # 实现其他 ModelInterface 要求的方法
    def train(self, data):
        # OpenAI 模型通常不需要本地训练，可以留空或抛出 NotImplementedError
        raise NotImplementedError("OpenAI models do not support local training")

    def evaluate(self, test_data):
        # 实现评估逻辑，如果适用的话
        pass

    # 可以添加其他必要的方法