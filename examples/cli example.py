from src.interfaces.cli import run_cli
from src.core.ai_assistant import AIAssistant
from src.dialogue_manager import DialogueManager


def main():
    assistant = AIAssistant()
    dialogue_manager = DialogueManager()
    run_cli(assistant, dialogue_manager)


if __name__ == "__main__":
    main()
