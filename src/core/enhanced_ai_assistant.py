import re
import logging
import sys
import json
import langdetect
import base64
from io import StringIO
from deep_translator import GoogleTranslator
from cryptography.fernet import Fernet
from deep_translator import GoogleTranslator
from langdetect import detect
from typing import Any, Dict, List
from src.core.ai_assistant import AIAssistant
from src.core.langchain import LangChainAgent
from src.utils.error_handler import error_handler
from src.utils.exceptions import InputValidationError, ModelError
from src.utils.exceptions import AIAssistantException
from cryptography.fernet import Fernet, InvalidToken
from src.core.language_model import LanguageModel

# 设置日志记录器
logger = logging.getLogger(__name__)

class EnhancedAIAssistant:
    def __init__(self):
        self.ai_assistant = AIAssistant()
        self.lang_chain_agent = LangChainAgent()
        self.cipher_suite = Fernet(Fernet.generate_key())
        self.language_model = LanguageModel()  # 确保已导入 LanguageModel
    
    @error_handler
    def detect_language(self, text: str) -> str:
        try:
            return detect(text)
        except Exception as e:
            logger.error(f"Language detection error: {str(e)}")
            return 'en'  # 默认返回英语
    
    @error_handler
    def translate(self, text: str, target_lang: str) -> str:
        try:
            translator = GoogleTranslator(source='auto', target=target_lang)
            return translator.translate(text)
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text  # 如果翻译失败，返回原文
        
    @error_handler
    def process_input(self, input_data: Any, source_language: str = None) -> str:
        if isinstance(input_data, str):
            if not source_language:
                source_language = self.detect_language(input_data)
            
            if source_language != 'en':
                input_data = self.translate(input_data, source_language, 'en')

        response = self.ai_assistant.process_input(input_data)

        if source_language != 'en':
            response = self.translate(response, 'en', source_language)

        return response

    @error_handler
    def generate_response(self, prompt: str, language: str = 'en') -> str:
        if language != 'en':
            prompt = self.translate(prompt, language, 'en')
        
        response = self.lang_chain_agent.run_generation_task(prompt)
        
        if language != 'en':
            response = self.translate(response, 'en', language)
        
        return response

    @error_handler
    def summarize(self, text: str) -> str:
        try:
            detected_lang = self.detect_language(text)
            
            # 如果不是英语，先翻译成英语
            if detected_lang != 'en':
                text = self.translate(text, 'en')

            # 使用 AIAssistant 的 summarize 方法
            summary = self.ai_assistant.summarize(text)

            # 如果原文不是英语，将摘要翻译回原语言
            if detected_lang != 'en':
                summary = self.translate(summary, detected_lang)

            return summary
        except Exception as e:
            logger.error(f"Summarization error: {str(e)}")
            return f"生成摘要失败: {str(e)}"

    def detect_language(self, text: str) -> str:
        # 这里应该实现实际的语言检测逻辑
        # 现在我们简单地假设所有输入都是中文
        return 'zh'

    def translate(self, text: str, target_lang: str) -> str:
        try:
            if target_lang == 'zh':
                target_lang = 'zh-CN'  # 使用简体中文
            translator = GoogleTranslator(source='auto', target=target_lang)
            return translator.translate(text)
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text  # 如果翻译失败，返回原文
    
    @error_handler
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        try:
            sentiment_prompt = f"请分析以下文本的情感，并以JSON格式返回结果，包括positive、neutral和negative的比例：\n\n{text}"
            sentiment_response = self.language_model.generate_response(sentiment_prompt)
        
            logger.info(f"Raw sentiment response: {sentiment_response}")

            # 移除可能的 Markdown 代码块标记
            cleaned_response = re.sub(r'```json\s*|\s*```', '', sentiment_response)
            logger.info(f"Cleaned response: {cleaned_response}")
        
            sentiment = json.loads(cleaned_response)

            logger.info(f"Parsed sentiment: {sentiment}")

            # 确保所有必要的键都存在，并且值是有效的浮点数
            for key in ['positive', 'neutral', 'negative']:
                if key not in sentiment or not isinstance(sentiment[key], (int, float)):
                    sentiment[key] = 0.0
                else:
                    sentiment[key] = float(sentiment[key])

            # 确保三个值的总和为 1
            total = sum(sentiment.values())
            if total != 0:
                sentiment = {k: v / total for k, v in sentiment.items()}

            logger.info(f"Final sentiment: {sentiment}")
            return sentiment
        except Exception as e:
            logger.error(f"Error in analyze_sentiment: {str(e)}", exc_info=True)
            return {"error": str(e), "positive": 0.0, "neutral": 0.0, "negative": 0.0}
    
    @error_handler
    def plan_task(self, task_description: str) -> str:
        try:
            detected_lang = self.detect_language(task_description)
            if detected_lang != 'en':
                task_description = self.translate(task_description, 'en')

            plan = self.ai_assistant.plan_task(task_description)

            if detected_lang != 'en':
                plan = self.translate(plan, detected_lang)

            return plan
        except Exception as e:
            logger.error(f"Error in plan_task: {str(e)}")
            return f"任务规划失败: {str(e)}"
   
    @error_handler
    def extract_keywords(self, text: str) -> List[str]:
        try:
            prompt = f"请从以下文本中提取关键词，以逗号分隔：\n\n{text}"
            response = self.language_model.generate_response(prompt)
            return [keyword.strip() for keyword in response.split(',')]
        except Exception as e:
            logger.error(f"Error in extract_keywords: {str(e)}", exc_info=True)
            return []

    @error_handler
    def change_model(self, model_name: str) -> None:
        self.ai_assistant.change_model(model_name)
        # 如果 LangChainAgent 支持更改模型，可以取消下面的注释
        # if hasattr(self.lang_chain_agent, 'change_model'):
        #     self.lang_chain_agent.change_model(model_name)
        # else:
        #     print("警告：LangChainAgent 不支持更改模型")
   
    @error_handler
    def get_available_models(self) -> List[str]:
        ai_assistant_models = self.ai_assistant.get_available_models()
        # 如果 LangChainAgent 没有 get_available_models 方法，就只返回 ai_assistant_models
        return ai_assistant_models
   
    @error_handler
    def encrypt_sensitive_data(self, data: str) -> str:
        try:
            encrypted_data = self.cipher_suite.encrypt(data.encode())
            return base64.b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"加密数据时发生错误: {str(e)}")
            raise AIAssistantException("加密失败")
        
    @error_handler
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        try:
            decoded_data = base64.b64decode(encrypted_data)
            decrypted_data = self.cipher_suite.decrypt(decoded_data)
            return decrypted_data.decode()
        except (InvalidToken, base64.binascii.Error) as e:
            logger.error(f"解密数据时发生错误: {str(e)}")
            raise AIAssistantException("解密失败：无效的令牌或数据格式")
        except Exception as e:
            logger.error(f"解密数据时发生未知错误: {str(e)}")
            raise AIAssistantException("解密失败：未知错误")

    @error_handler
    def reinforcement_learning_action(self, state: Any) -> Any:
        return self.ai_assistant.reinforcement_learning_action(state)

    @error_handler
    def active_learning_sample(self, unlabeled_data: List[Any]) -> Any:
        return self.ai_assistant.active_learning_sample(unlabeled_data)

    @error_handler
    def clear_context(self) -> None:
        self.ai_assistant.clear_context()

    @error_handler
    def get_dialogue_context(self) -> List[Dict[str, str]]:
        return self.ai_assistant.context
    
    def execute_code(self, code: str, language: str = 'python') -> str:
        try:
            # 如果 AIAssistant 有 execute_code 方法，优先使用它
            if hasattr(self.ai_assistant, 'execute_code'):
                return self.ai_assistant.execute_code(code, language)
            
            # 否则，使用自己的实现
            if language.lower() != 'python':
                return f"暂不支持 {language} 语言的代码执行"

            import sys
            from io import StringIO

            old_stdout = sys.stdout
            redirected_output = sys.stdout = StringIO()

            exec(code, globals())

            sys.stdout = old_stdout
            result = redirected_output.getvalue()

            return result
        except Exception as e:
            return f"执行错误: {str(e)}"
    
    @error_handler
    def answer_question(self, question: str) -> str:
        try:
            prompt = f"请回答以下问题：\n\n{question}"
            response = self.language_model.generate_response(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Error in answer_question: {str(e)}", exc_info=True)
            return f"抱歉，我无法回答这个问题。错误：{str(e)}"