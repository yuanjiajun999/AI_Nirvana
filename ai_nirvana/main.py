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
        self.language_model = LanguageModel(self.config.get('model_name', 'gpt2'))
        self.code_executor = CodeExecutor()
        self.context_manager = ContextManager()
        self.security_manager = SecurityManager()
        self.logger = setup_logger('ai_nirvana', 'ai_nirvana.log')

    def process(self, input_text):
        self.logger.info(f"Processing input: {input_text}")
        context = self.context_manager.get_context()
        response = self.language_model.generate_response(input_text, context)
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
        cli.run_cli()
    elif interface == 'gui':
        gui.run_gui(ai_nirvana)
    elif interface == 'api':
        api.run_api()
    elif interface == 'sd_web':
        sd_web_controller.run_sd_web_controller()
    else:
        print(f"Unknown interface: {interface}")

if __name__ == "__main__":
    main()