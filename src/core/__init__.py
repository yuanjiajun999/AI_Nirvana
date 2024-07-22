from typing import List, Any  

# 使用相对导入  
from .active_learning import ActiveLearner  
from .digital_twin import DigitalTwin  
from .generative_ai import GenerativeAI  
from .intelligent_agent import IntelligentAgent  
from .lora import LoRAModel  
from .multimodal import MultimodalInterface  
from .privacy_enhancement import PrivacyEnhancement  
from .quantization import quantize_and_evaluate  
from .reinforcement_learning import ReinforcementLearningAgent  
from .semi_supervised_learning import SemiSupervisedTrainer  

# 定义 __all__ 变量，明确指定可以从这个模块导入的名称  
__all__: List[str] = [  
    "ActiveLearner",  
    "DigitalTwin",  
    "GenerativeAI",  
    "IntelligentAgent",  
    "LoRAModel",  
    "MultimodalInterface",  
    "PrivacyEnhancement",  
    "quantize_and_evaluate",  
    "ReinforcementLearningAgent",  
    "SemiSupervisedTrainer"  
]  

def get_available_models() -> List[str]:  
    """  
    返回所有可用模型的列表。  

    Returns:  
        List[str]: 可用模型的名称列表  
    """  
    return [  
        "ActiveLearner",  
        "DigitalTwin",  
        "GenerativeAI",  
        "IntelligentAgent",  
        "LoRAModel",  
        "MultimodalInterface",  
        "PrivacyEnhancement",  
        "ReinforcementLearningAgent",  
        "SemiSupervisedTrainer"  
    ]  

def get_model(model_name: str) -> Any:  
    """  
    根据模型名称返回相应的模型类。  

    Args:  
        model_name (str): 模型的名称  

    Returns:  
        Any: 对应的模型类  

    Raises:  
        ValueError: 如果提供的模型名称无效  
    """  
    models = {  
        "ActiveLearner": ActiveLearner,  
        "DigitalTwin": DigitalTwin,  
        "GenerativeAI": GenerativeAI,  
        "IntelligentAgent": IntelligentAgent,  
        "LoRAModel": LoRAModel,  
        "MultimodalInterface": MultimodalInterface,  
        "PrivacyEnhancement": PrivacyEnhancement,  
        "ReinforcementLearningAgent": ReinforcementLearningAgent,  
        "SemiSupervisedTrainer": SemiSupervisedTrainer  
    }  
    
    if model_name not in models:  
        raise ValueError(f"Invalid model name: {model_name}")  
    
    return models[model_name]