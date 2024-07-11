from ai_nirvana.core.language_model import LanguageModel
from ai_nirvana.core.code_executor import CodeExecutor
from ai_nirvana.core.context_manager import ContextManager
from ai_nirvana.utils.config import Config
from ai_nirvana.utils.security import SecurityManager
from ai_nirvana.utils.logger import setup_logger
from ai_nirvana.interfaces import cli, gui, api

class AINirvana:
    def __init__(self):
        self.config = Config()
        self.language_model = LanguageModel(self.config.get('model_name', 'gpt2-large'), use_gpu=True)  # 使用更大的模型，并尝试使用GPU
        self.code_executor = CodeExecutor()
        self.context_manager = ContextManager()
        self.security_manager = SecurityManager()
        self.logger = setup_logger('ai_nirvana', 'ai_nirvana.log')
        self.system_prompt = (
            "You are AI Nirvana, an advanced AI assistant designed to provide accurate, relevant, and helpful information. "
            "Your responses should be direct, concise, and on-topic. When answering questions, provide clear explanations. "
            "If asked about a topic you're unsure of, admit your uncertainty. "
            "When asked to write code, provide well-structured, functioning code with clear comments. "
            "Avoid repetition, stay focused on the specific question asked, and strive for high-quality, informative responses."
        )

    def process(self, input_text):
        self.logger.info(f"Processing input: {input_text}")
        context = self.context_manager.get_context()
        prompt = f"{self.system_prompt}\n\nContext: {context}\n\nHuman: {input_text}\nAI: Let me provide a clear, concise, and accurate answer to your question or request:\n"

        max_attempts = 3
        for _ in range(max_attempts):
            response = self.language_model.generate_response(prompt, max_length=200)  # 增加最大长度

            # 改进的质量检查
            if len(response.split()) > 20 and '.' in response:
                break
        else:
            response = "I apologize, but I'm having trouble generating a proper response. Could you please rephrase your question or request?"

        self.context_manager.add_to_context(f"Human: {input_text}")
        self.context_manager.add_to_context(f"AI: {response}")

        if "```python" in response:
            code = response.split("```python")[1].split("```")[0].strip()
            stdout, stderr = self.code_executor.execute_code(code, 'python')
            response += f"\n\nExecution result:\n{stdout}\n\nErrors:\n{stderr}"

        self.logger.info(f"Generated response: {response}")
        return response

def main():
    ai_nirvana = AINirvana()
    interface = ai_nirvana.config.get('interface', 'cli')

    if interface == 'cli':
        cli.run_cli(ai_nirvana)
    elif interface == 'gui':
        gui.run_gui(ai_nirvana)
    elif interface == 'api':
        api.run_api(ai_nirvana)
    elif interface == 'sd_web':
        print("sd_web interface not implemented yet. Falling back to CLI.")
        cli.run_cli(ai_nirvana)
    else:
        print(f"Unknown interface: {interface}")
        cli.run_cli(ai_nirvana)  # 默认使用 CLI 接口

if __name__ == "__main__":
    main()