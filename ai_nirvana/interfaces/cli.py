from ai_nirvana.core.ai_nirvana import AINirvana

def run_cli():
    ai_nirvana = AINirvana()
    print("Welcome to AI Nirvana. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = ai_nirvana.process(user_input)
        print(f"AI: {response}")