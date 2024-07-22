from abc import ABC, abstractmethod


class ModelInterface(ABC):
    @abstractmethod
    def generate_response(self, prompt):
        pass

    @abstractmethod
    def summarize(self, text):
        pass