import openai
import os
from dotenv import load_dotenv

load_dotenv()

class LanguageModel:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate_response(self, prompt, context=""):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in generating response: {e}")
            return "Sorry, I couldn't generate a response at this time."