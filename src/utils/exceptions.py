class AIAssistantException(Exception):
    """Custom exception class for AI Assistant related errors."""
    pass

class InputValidationError(AIAssistantException):
    pass

class ModelError(AIAssistantException):
    pass