import logging

class AIAssistant:
    def __init__(self, model):
        self.model = model

    def generate_response(self, prompt):
        try:
            response = self.model.generate_response(prompt)
            logging.info(f"User input: {prompt}")
            logging.info(f"System response: {response}")
            return response
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logging.error(error_msg)
            return "抱歉，生成回答时出现了错误。请稍后再试。"

    def summarize(self, text):
        try:
            summary = self.model.summarize(text)
            logging.info(f"Summarization request: {text[:100]}...")
            logging.info(f"Summary: {summary}")
            return summary
        except Exception as e:
            error_msg = f"Error generating summary: {str(e)}"
            logging.error(error_msg)
            return "抱歉，生成摘要时出现了错误。请稍后再试。"