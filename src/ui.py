import colorama
from colorama import Fore, Style
from src.dialogue_manager import DialogueManager

def print_user_input(user_input):
    print(f"{Fore.BLUE}User: {user_input}{Style.RESET_ALL}")

def print_assistant_response(response):
    print(f"{Fore.GREEN}Assistant: {response}{Style.RESET_ALL}")

def print_dialogue_context(context):
    print(f"{Fore.YELLOW}Dialogue Context:{Style.RESET_ALL}\n{context}")

def main():
    dialogue_manager = DialogueManager(max_history=5)

    while True:
        user_input = input("\nEnter your question or text (or 'quit' to exit): ")

        if user_input.lower() == 'quit':
            print("Thank you for using the AI Nirvana assistant. Goodbye!")
            break

        response = generate_response(user_input)
        print_user_input(user_input)
        print_assistant_response(response)

        dialogue_manager.add_to_history(user_input, response)
        print_dialogue_context(dialogue_manager.get_dialogue_context())