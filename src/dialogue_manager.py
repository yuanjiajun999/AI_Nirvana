from collections import deque

class DialogueManager:
    def __init__(self, max_history=5):
        self.history = deque(maxlen=max_history)

    def add_to_history(self, user_input, assistant_response):
        self.history.append((user_input, assistant_response))

    def get_dialogue_context(self):
        return "\n".join([f"User: {user_input}\nAssistant: {response}" for user_input, response in self.history])

    def clear_history(self):
        self.history.clear()
        return "对话历史已清除。"