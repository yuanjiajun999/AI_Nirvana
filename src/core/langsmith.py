import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import Client
from functools import lru_cache

load_dotenv()

class LangSmith:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-0125",
            openai_api_key=os.getenv("API_KEY"),
            openai_api_base=os.getenv("API_BASE")
        )
        self.client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

    @lru_cache(maxsize=100)
    def _invoke_llm(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content.strip()

    def generate_code(self, prompt: str) -> str:
        try:
            return self._invoke_llm(f"Generate Python code for: {prompt}")
        except Exception as e:
            print(f"Error in generate_code: {str(e)}")
            return "Sorry, I couldn't generate the code."

    def refactor_code(self, code: str) -> str:
        try:
            return self._invoke_llm(f"Refactor this Python code:\n\n{code}\n\nRefactored code:")
        except Exception as e:
            print(f"Error in refactor_code: {str(e)}")
            return "Sorry, I couldn't refactor the code."

    def translate_text(self, text: str, target_lang: str) -> str:
        try:
            return self._invoke_llm(f"Translate the following text to {target_lang}:\n\n{text}\n\nTranslation:")
        except Exception as e:
            print(f"Error in translate_text: {str(e)}")
            return "Sorry, I couldn't translate the text."