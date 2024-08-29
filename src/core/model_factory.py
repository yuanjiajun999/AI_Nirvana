from typing import Dict, Type, List
from .model_interface import ModelInterface

class ModelFactory:
    _models: Dict[str, Type[ModelInterface]] = {}

    @classmethod
    def register_model(cls, name: str, model_class: Type[ModelInterface]):
        cls._models[name] = model_class

    @classmethod
    def create_model(cls, name: str, **kwargs) -> ModelInterface:
        if name not in cls._models:
            raise ValueError(f"Model {name} not registered")
        return cls._models[name](**kwargs)

    @classmethod
    def get_available_models(cls) -> List[str]:
        return list(cls._models.keys())

    
    