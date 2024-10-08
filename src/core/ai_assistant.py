import re
import sys
from io import StringIO
print(sys.path)
import logging
from typing import Any, Dict, List, Optional
import json
from src.core.model_factory import ModelFactory 
from src.core.language_model import LanguageModel
from src.core.security import SecurityManager
from src.core.knowledge_base import KnowledgeBase
from src.core.multimodal import MultimodalInterface
from src.core.reasoning import ReasoningEngine
from src.utils.error_handler import error_handler, logger
from src.utils.exceptions import AIAssistantException, InputValidationError, ModelError

class AIAssistant:
    """
    AI 助手类，提供多模态输入处理、生成回应、文本摘要、情感分析等功能，
    并支持上下文管理、安全检查、知识检索和推理。

    Attributes:
        language_model: 用于生成回应、摘要和情感分析的语言模型。
        security_manager: 用于进行安全检查和数据加密的安全管理器。
        knowledge_base: 用于知识检索的知识库。
        multimodal_interface: 用于处理多模态输入的接口。
        reasoning_engine: 用于推理的引擎。
        context: 存储对话历史的列表。
        max_context_length: 保存的最大上下文长度。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, api_client: Optional[Any] = None, model_name: str = "gpt-3.5-turbo", max_context_length: int = 5):
        """
        初始化 AI 助手。

        Args:
            config: 系统配置对象。
            api_client: API 客户端实例。
            model_name: 要使用的模型名称。
            max_context_length: 保存的最大上下文长度。
        """
        self.language_model = ModelFactory.create_model(
            "LanguageModel", 
            config=config, 
            api_client=api_client, 
            model_name=model_name
        )
        self.security_manager = SecurityManager()
        self.knowledge_base = KnowledgeBase(config, api_client)  # 传递 config 和 api_client
        self.multimodal_interface = MultimodalInterface()  # 不需要参数
        self.reasoning_engine = ReasoningEngine()
        self.context: List[Dict[str, str]] = []
        self.max_context_length = max_context_length
        logger.info(f"AI Assistant initialized with model: {model_name}")

    @error_handler
    def process_input(self, input_data: Any) -> str:
        """
        处理输入数据并生成回应。

        Args:
            input_data: 用户输入的数据，可以是多模态的。

        Returns:
            str: 生成的回应。

        Raises:
            InputValidationError: 如果输入不安全。
            ModelError: 如果生成回应时发生错误。
        """
        try:
            processed_input = self.multimodal_interface.process(input_data)
            
            if not self.security_manager.is_safe(processed_input):
                raise InputValidationError("Input rejected due to security concerns.")
            
            relevant_knowledge = self.knowledge_base.retrieve(processed_input)
            reasoning_result = self.reasoning_engine.reason(processed_input, relevant_knowledge)
            response = self.language_model.generate_response(reasoning_result)
            
            self._update_context("user", str(input_data))
            self._update_context("assistant", response)
            
            logger.info(f"User input: {str(input_data)[:50]}...")
            logger.info(f"System response: {response[:50]}...")
            
            return response
        except Exception as e:
            logger.error(f"Error in process_input: {str(e)}")
            raise

    @error_handler
    def generate_response(self, prompt: str) -> str:
        """
        根据给定的提示和上下文生成回应。

        Args:
            prompt: 用户输入的提示。

        Returns:
            str: 生成的回应。

        Raises:
            InputValidationError: 如果输入不安全。
            ModelError: 如果生成回应时发生错误。
        """
        def generate_response(self, prompt: str) -> str:
            try:
                response = self.language_model.generate(prompt)
                # 尝试提取JSON部分
                json_pattern = r'\{[^{}]*\}'
                match = re.search(json_pattern, response)
                if match:
                    return match.group()
                else:
                    return response
            except Exception as e:
                logger.error(f"Error in generate_response: {str(e)}")
                return str(e)
   
    @error_handler
    def summarize(self, text: str) -> str:
        """
        生成给定文本的摘要。

        Args:
            text: 需要摘要的文本。

        Returns:
            str: 生成的摘要。

        Raises:
            InputValidationError: 如果输入不安全。
            ModelError: 如果生成摘要时发生错误。
        """
        try:
            if not self.security_manager.is_safe_code(text):
                raise InputValidationError("Unsafe code detected in text")

            summary_prompt = f"Please summarize the following text concisely in no more than 100 words:\n\n{text}"
            summary = self.language_model.generate_response(summary_prompt)

            logger.info(f"Summarization request: {text[:50]}...")
            logger.info(f"Summary: {summary[:50]}...")

            return summary
        except Exception as e:
            logger.error(f"Error in summarize: {str(e)}")
            raise

    @error_handler
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        分析给定文本的情感。

        Args:
            text: 需要分析情感的文本。

        Returns:
            Dict[str, float]: 包含情感分析结果的字典。
            例如：{'positive': 0.8, 'neutral': 0.15, 'negative': 0.05}

        Raises:
            InputValidationError: 如果输入不安全。
            ModelError: 如果情感分析时发生错误。
        """
        try:
            sentiment_prompt = f"请分析以下文本的情感，并以JSON格式返回结果，包括positive、neutral和negative的比例：\n\n{text}"
            sentiment_response = self.language_model.generate_response(sentiment_prompt)
        
            # 尝试解析JSON响应
            try:
                sentiment = json.loads(sentiment_response)
            except json.JSONDecodeError:
                # 如果无法解析JSON，尝试提取最后一个有效的JSON对象
                import re
                json_pattern = r'\{(?:[^{}]|(?R))*\}'
                matches = re.findall(json_pattern, sentiment_response)
                if matches:
                    sentiment = json.loads(matches[-1])
                else:
                    raise ValueError("无法解析情感分析结果")

            # 确保所有必要的键都存在
            for key in ['positive', 'neutral', 'negative']:
                if key not in sentiment:
                    sentiment[key] = 0.0

            logger.info(f"Sentiment analysis request: {text[:50]}...")
            logger.info(f"Sentiment analysis result: {sentiment}")

            return sentiment
        except Exception as e:
            logger.error(f"Error in analyze_sentiment: {str(e)}")
            return {"error": str(e)}
   
    def _update_context(self, role: str, content: str) -> None:
        """
        更新对话上下文。

        Args:
            role: 消息的角色（'user' 或 'assistant'）。
            content: 消息内容。
        """
        self.context.append({"role": role, "content": content})
        if len(self.context) > self.max_context_length:
            self.context.pop(0)

    def clear_context(self) -> None:
        """
        清除当前的对话上下文。
        """
        self.context.clear()
        logger.info("Context cleared.")

    @error_handler
    def change_model(self, model_name: str) -> None:
        if model_name in self.get_available_models():
            self.model_name = model_name
            # 如果需要，在这里重新初始化模型
            print(f"模型已更改为: {model_name}")
        else:
            raise ValueError(f"不支持的模型: {model_name}")

    @error_handler
    def get_available_models(self) -> List[str]:
        """
        获取可用的语言模型列表。

        Returns:
            List[str]: 可用模型的列表。

        Raises:
            ModelError: 如果获取模型列表失败。
        """
        try:
            models = self.language_model.get_available_models()
            logger.info(f"Available models: {models}")
            return models
        except Exception as e:
            logger.error(f"Error in get_available_models: {str(e)}")
            raise ModelError(f"Failed to get available models: {str(e)}")

    @error_handler
    def encrypt_sensitive_data(self, data: str) -> str:
        """
        加密敏感数据。

        Args:
            data: 需要加密的数据。

        Returns:
            str: 加密后的数据。

        Raises:
            AIAssistantException: 如果加密过程中发生错误。
        """
        try:
            encrypted_data = self.security_manager.encrypt_sensitive_data(data)
            logger.info("Data encrypted successfully")
            return encrypted_data
        except Exception as e:
            logger.error(f"Error in encrypt_sensitive_data: {str(e)}")
            raise AIAssistantException(f"Failed to encrypt data: {str(e)}")

    @error_handler
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """
        解密敏感数据。

        Args:
            encrypted_data: 需要解密的数据。

        Returns:
            str: 解密后的数据。

        Raises:
            AIAssistantException: 如果解密过程中发生错误。
        """
        try:
            decrypted_data = self.security_manager.decrypt_sensitive_data(encrypted_data)
            logger.info("Data decrypted successfully")
            return decrypted_data
        except Exception as e:
            logger.error(f"Error in decrypt_sensitive_data: {str(e)}")
            raise AIAssistantException(f"Failed to decrypt data: {str(e)}")

    @error_handler
    def execute_code(self, code: str, language: str = 'python') -> str:
        if language.lower() != 'python':
            return f"暂不支持 {language} 语言的代码执行"

        try:
            # 捕获标准输出
            old_stdout = sys.stdout
            redirected_output = sys.stdout = StringIO()

            # 执行代码
            exec(code, globals())

            # 恢复标准输出
            sys.stdout = old_stdout

            # 获取执行结果
            result = redirected_output.getvalue()

            return result if result else "代码执行成功，但没有输出。"
        except Exception as e:
            return f"执行错误: {str(e)}"
   
    @error_handler
    def plan_task(self, task_description: str) -> str:
        """
        为给定的任务生成计划。

        Args:
            task_description: 任务描述。

        Returns:
            str: 生成的任务计划。

        Raises:
            ModelError: 如果生成计划时发生错误。
        """
        try:
            plan_prompt = f"Generate a step-by-step plan for the following task:\n\n{task_description}"
            plan = self.language_model.generate_response(plan_prompt)
            logger.info(f"Task planning request: {task_description[:50]}...")
            logger.info(f"Generated plan: {plan[:50]}...")
            return plan
        except Exception as e:
            logger.error(f"Error in plan_task: {str(e)}")
            raise ModelError(f"Failed to generate task plan: {str(e)}")

    @error_handler
    def reinforcement_learning_action(self, state: Any) -> Any:
        """
        使用强化学习选择动作。

        Args:
            state: 当前状态。

        Returns:
            Any: 选择的动作。

        Raises:
            ModelError: 如果强化学习过程中发生错误。
        """
        try:
            action = self.reasoning_engine.reinforcement_learning(state)
            logger.info(f"Reinforcement learning action for state: {state}")
            logger.info(f"Selected action: {action}")
            return action
        except Exception as e:
            logger.error(f"Error in reinforcement_learning_action: {str(e)}")
            raise ModelError(f"Failed to perform reinforcement learning: {str(e)}")

    @error_handler
    def active_learning_sample(self, unlabeled_data: List[Any]) -> Any:
        """
        使用主动学习选择样本。

        Args:
            unlabeled_data: 未标记的数据列表。

        Returns:
            Any: 选择的样本。

        Raises:
            ModelError: 如果主动学习过程中发生错误。
        """
        try:
            sample = self.reasoning_engine.active_learning(unlabeled_data)
            logger.info(f"Active learning sample selection from {len(unlabeled_data)} unlabeled data points")
            logger.info(f"Selected sample: {sample}")
            return sample
        except Exception as e:
            logger.error(f"Error in active_learning_sample: {str(e)}")
            raise ModelError(f"Failed to perform active learning: {str(e)}")
    
    def detect_language(self, text: str) -> str:
        # 这里应该实现实际的语言检测逻辑
        # 现在我们简单地假设所有输入都是中文
        return 'zh'

    def translate(self, text: str, target_lang: str) -> str:
        # 这里应该实现实际的翻译逻辑
        # 现在我们简单地返回原文
        return text    