class ContextManager:
    def __init__(self, max_context_length=1000):
        self.context = []
        self.max_context_length = max_context_length

    def add_to_context(self, message):
        self.context.append(message)
        if len(self.context) > self.max_context_length:
            self.context.pop(0)

    def get_context(self):
        return "\n".join(self.context)

    def clear_context(self):
        self.context = []