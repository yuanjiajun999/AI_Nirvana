from typing import Any, List, Type, Dict  

# 使用相对导入  
from .active_learning import ActiveLearner  
from .digital_twin import DigitalTwin  
from .generative_ai import GenerativeAI  
from .intelligent_agent import IntelligentAgent  
from .lora import LoRAModel  
from .multimodal import MultimodalInterface  
from .privacy_enhancement import PrivacyEnhancement  
from .quantization import quantize_and_evaluate  
from .reinforcement_learning import (  
    DQNAgent, A2CAgent, PPOAgent, SACAgent, TD3Agent, DDPGAgent,  
    MultiAgentRL, HierarchicalRL, CuriosityDrivenRL, MetaLearningAgent,  
    create_environment, train_agent, evaluate_agent,  
    plot_learning_curve, save_agent, load_agent, parallel_train_agents  
)  
from .semi_supervised_learning import SemiSupervisedDataset, AdvancedSemiSupervisedTrainer  
from .ai_assistant import AIAssistant  
from .language_model import LanguageModel  
from .model_interface import ModelInterface  
from .model_factory import ModelFactory  
from .auto_feature_engineering import AutoFeatureEngineer  # 新增  
from .vector_store import VectorStore
from .langsmith import LangSmithIntegration
from .langgraph import LangGraph

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
    "DQNAgent",  
    "A2CAgent",  
    "PPOAgent",  
    "SACAgent",  
    "TD3Agent",  
    "DDPGAgent",  
    "MultiAgentRL",  
    "HierarchicalRL",  
    "CuriosityDrivenRL",  
    "MetaLearningAgent",  
    "SemiSupervisedDataset",  
    "AdvancedSemiSupervisedTrainer",  
    "AIAssistant",  
    "LanguageModel",  
    "ModelInterface",  
    "ModelFactory",  
    "create_environment",  
    "train_agent",  
    "evaluate_agent",  
    "plot_learning_curve",  
    "save_agent",  
    "load_agent",  
    "parallel_train_agents",  
    "AutoFeatureEngineer"  # 新增 
    "VectorStore",  # 新增
    "ExtendedNetworkxEntityGraph",
    "LangGraph",     # 新增
    "LangSmithIntegration", # 新增
]  

def get_available_model_classes() -> Dict[str, Type[ModelInterface]]:  
    """  
    返回所有可用模型类的字典。  

    Returns:  
        Dict[str, Type[ModelInterface]]: 模型名称和对应的模型类的字典  
    """  
    return {  
        "ActiveLearner": ActiveLearner,  
        "DigitalTwin": DigitalTwin,  
        "GenerativeAI": GenerativeAI,  
        "IntelligentAgent": IntelligentAgent,  
        "LoRAModel": LoRAModel,  
        "MultimodalInterface": MultimodalInterface,  
        "PrivacyEnhancement": PrivacyEnhancement,  
        "DQNAgent": DQNAgent,  
        "A2CAgent": A2CAgent,  
        "PPOAgent": PPOAgent,  
        "SACAgent": SACAgent,  
        "TD3Agent": TD3Agent,  
        "DDPGAgent": DDPGAgent,  
        "MultiAgentRL": MultiAgentRL,  
        "HierarchicalRL": HierarchicalRL,  
        "CuriosityDrivenRL": CuriosityDrivenRL,  
        "MetaLearningAgent": MetaLearningAgent,  
        "AdvancedSemiSupervisedTrainer": AdvancedSemiSupervisedTrainer,  
        "LangSmithIntegration":LangSmithIntegration, # 新增
        "AutoFeatureEngineer": AutoFeatureEngineer,
        "VectorStore": VectorStore,  # 新增
        "LangGraph": LangGraph       # 新增
    }

def get_model(model_name: str) -> Type[ModelInterface]:  
    """  
    根据模型名称返回相应的模型类。  

    Args:  
        model_name (str): 模型的名称  

    Returns:  
        Type[ModelInterface]: 对应的模型类  

    Raises:  
        ValueError: 如果提供的模型名称无效  
    """  
    models = get_available_model_classes()  
    if model_name not in models:  
        raise ValueError(f"Invalid model name: {model_name}")  
    return models[model_name]  

# 使用ModelFactory注册所有模型  
for model_name, model_class in get_available_model_classes().items():  
    ModelFactory.register_model(model_name, model_class)  

def create_model(model_name: str, **kwargs: Any) -> ModelInterface:  
    """  
    使用ModelFactory创建并返回指定的模型实例。  

    Args:  
        model_name (str): 模型的名称  
        **kwargs: 传递给模型构造函数的额外参数  

    Returns:  
        ModelInterface: 创建的模型实例  

    Raises:  
        ValueError: 如果提供的模型名称无效  
    """  
    return ModelFactory.create_model(model_name, **kwargs)