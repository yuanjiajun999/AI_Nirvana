class AIAssistantException(Exception):
    pass

class InputValidationError(AIAssistantException):
    pass

class ModelError(AIAssistantException):
    pass