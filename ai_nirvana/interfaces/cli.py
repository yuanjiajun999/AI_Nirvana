from ai_nirvana.core.language_model import LanguageModel

def run_cli(ai_nirvana):
    print("Welcome to AI Nirvana. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = ai_nirvana.process(user_input)
        print(f"AI: {response}")

if __name__ == "__main__":
    from ai_nirvana.main import AINirvana
    ai_nirvana = AINirvana()
    run_cli(ai_nirvana)