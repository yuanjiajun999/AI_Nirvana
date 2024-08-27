import os
import io
import sys
import ast
import argparse
import logging
import time
import networkx as nx
from langdetect import detect
from openai import OpenAI 
import scipy.optimize as opt
import numpy as np
from tqdm import tqdm
import featuretools as ft
import concurrent.futures
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from typing import Dict, Any, List
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from tests.complex_text_analysis import test_complex_text_analysis
from tests.model_comparison import test_model_comparison
from tests.code_execution_safety import test_code_execution_safety
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd 
from src.core.code_executor import CodeExecutor
from src.core.model_factory import ModelFactory
from src.core.language_model import LanguageModel  # 假设 LanguageModel 是模型实现类
from cryptography.fernet import InvalidToken
from src.utils.exceptions import AIAssistantException
from src.core.enhanced_ai_assistant import EnhancedAIAssistant

# 调整路径以便于模块导入  
current_dir = os.path.dirname(os.path.abspath(__file__))  
parent_dir = os.path.dirname(current_dir)  
sys.path.append(parent_dir)

from src.config import Config
from src.core.api_client import ApiClient
from src.core.ai_assistant import AIAssistant
from src.core.generative_ai import GenerativeAI
from src.core.multimodal import MultimodalInterface
from src.utils.error_handler import error_handler, logger, AIAssistantException
from src.utils.security import SecurityManager
from src.dialogue_manager import DialogueManager
from src.ui import print_user_input, print_assistant_response, print_dialogue_context, print_sentiment_analysis

# 新增导入
from src.core.model_factory import ModelFactory
from src.core.privacy_enhancement import PrivacyEnhancement
from src.core.model_interpretability import ModelInterpreter
from src.core.semi_supervised_learning import AdvancedSemiSupervisedTrainer
from src.core.reinforcement_learning import DQNAgent
from src.core.knowledge_base import KnowledgeBase
from src.core.enhanced_ai_assistant import EnhancedAIAssistant  # 新增导入
from src.core.active_learning import ActiveLearner
from src.core.auto_feature_engineering import AutoFeatureEngineer
from src.core.digital_twin import DigitalTwin
from sklearn.ensemble import IsolationForest
from src.core.langgraph import LangGraph
from src.command_data import AVAILABLE_COMMANDS
from src.help_info import get_help

logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__) 

# 加载 .env 文件
load_dotenv()
print("API_KEY:", os.getenv("API_KEY"))
print("API_BASE:", os.getenv("API_BASE"))
print("MODEL_NAME:", os.getenv("MODEL_NAME"))

class AINirvana:
    def __init__(self, config: Config):
        # 注册模型
        ModelFactory.register_model("LanguageModel", LanguageModel)
        
        self.code_executor = CodeExecutor()
        self.config = config
        self.api_client = ApiClient(config.api_key)  # 使用 ApiClient 替代直接的 LLM 对象   
        self.model_name = os.getenv('MODEL_NAME', 'gpt-3.5-turbo')  # 使用环境变量中的 MODEL_NAME
        self.max_context_length = self.config.get('max_context_length', 5)
        self.model = ModelFactory.create_model("LanguageModel", model_name=self.model_name)
        self.assistant = EnhancedAIAssistant()  # 使用新的EnhancedAIAssistant
        self.generative_ai = GenerativeAI()
        self.multimodal_interface = MultimodalInterface()
        self.dialogue_manager = DialogueManager(max_history=self.max_context_length)
        self.security_manager = SecurityManager()
        self.privacy_enhancer = PrivacyEnhancement()
        self.variable_state = {}
        self.knowledge_base = KnowledgeBase()
        self.active_learner = None  # 初始化为 None
        self.accuracy_history = None
        self.feature_engineer = None  # 初始化为 None
        self.digital_twin = None  # 初始时未创建
        self.lang_graph = LangGraph()
        self.initialize_modules()  # 自动检测并初始化模块
        print(f"AINirvana instance created with model: {self.model_name}")

    @error_handler
    def process(self, input_text: Any) -> str:
        if not isinstance(input_text, str):
            raise ValueError("输入必须是字符串类型")
        if not self.security_manager.is_safe_code(input_text):
            raise AIAssistantException("检测到不安全的输入")
        language = self.assistant.detect_language(input_text)
        processed_input = self.multimodal_interface.process(input_text)
        response = self.assistant.process_input(processed_input, language)
        self.dialogue_manager.add_to_history(input_text, response)
        return response
   
    def summarize(self, text: str) -> str:
        return self.assistant.summarize(text)
        
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        return self.assistant.analyze_sentiment(text)

    def change_model(self, model_name: str) -> None:
        self.assistant.change_model(model_name)
        self.model_name = model_name  # 更新 AINirvana 实例的 model_name
        print(f"模型已更改为 {model_name}。")

    def add_knowledge(self, key: str, value: Any) -> None:
        self.knowledge_base.add_knowledge(key, value)

    def get_knowledge(self, key: str) -> Any:
        return self.knowledge_base.get_knowledge(key)

    def list_all_knowledge(self) -> Dict[str, Any]:
        return self.knowledge_base.list_all_knowledge()

    def update_knowledge(self, key: str, value: Any) -> None:
        self.knowledge_base.update_knowledge(key, value)

    def delete_knowledge(self, key: str) -> None:
        self.knowledge_base.delete_knowledge(key)

    def search_knowledge(self, query: str) -> Dict[str, Any]:
        return {k: v for k, v in self.knowledge_base.list_all_knowledge().items() 
                if query.lower() in k.lower() or (isinstance(v, str) and query.lower() in v.lower())}

    @error_handler
    def get_available_models(self) -> List[str]:
        try:
            models = self.assistant.get_available_models()
            if not models:
                logger.warning("没有可用的模型")
                return []
            return models
        except Exception as e:
            logger.error(f"获取可用模型时发生错误: {str(e)}")
            return []

    def update_variable_state(self, new_vars: dict):
        self.variable_state.update(new_vars)

    def get_variable_state(self) -> str:
        if not self.variable_state:
            return "当前没有定义的变量。"
        return "\n".join(f"{var} = {value}" for var, value in self.variable_state.items() if not var.startswith('__'))
    
    def is_unsafe_code(self, code: str) -> bool:  
        # 这里实现代码安全性检查的逻辑  
        unsafe_patterns = [  
            "import os",  
            "import subprocess",  
            "open(",  
            "import socket",  
            "__import__"  
        ]  
        return any(pattern in code for pattern in unsafe_patterns)  

    def execute_code(self, code: str) -> str:
        if self.is_unsafe_code(code):
            return "执行被阻止：不安全的代码", {}

        # 捕获标准输出和标准错误
        stdout = io.StringIO()
        stderr = io.StringIO()
        sys.stdout = stdout
        sys.stderr = stderr

        try:
            # 执行代码
            local_vars = {}
            exec(code, globals(), local_vars)
            output = stdout.getvalue()
            error = stderr.getvalue()

            if output:
                result = f"执行结果:\n{output}"
            elif error:
                result = f"执行过程中出现错误:\n{error}"
            else:
                # 如果没有输出，显示所有定义的变量
                var_output = []
                for var, value in local_vars.items():
                    if not var.startswith('__'):
                        var_output.append(f"{var} = {value}")
                if var_output:
                    result = "定义的变量:\n" + "\n".join(var_output)
                else:
                    # 如果没有定义新变量，尝试获取最后一个表达式的值
                    last_line = code.strip().split('\n')[-1]
                    try:
                        result = eval(last_line, globals(), local_vars)
                        result = f"最后一个表达式的值: {result}"
                    except:
                        result = "代码执行成功，但没有输出或定义新变量。"

            return result, local_vars

        except Exception as e:
            return f"代码执行时发生错误: {str(e)}", {}

        finally:
            # 恢复标准输出和标准错误
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        
    @error_handler
    def validate_code(self, code: str) -> bool:
        return self.code_executor.validate_code(code)

    @error_handler
    def get_supported_languages(self) -> List[str]:
        return self.code_executor.get_supported_languages()
    
    def encrypt_sensitive_data(self, data: str) -> str:
        return self.assistant.encrypt_sensitive_data(data)

    @error_handler
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        try:
            decrypted = self.assistant.decrypt_sensitive_data(encrypted_data)
            return f"解密结果: {decrypted}"
        except AIAssistantException as e:
            return f"解密失败: {str(e)}"
        
    def preprocess_data(self, data: str) -> pd.DataFrame:
        return pd.DataFrame([data.split()])

    def explain_model(self, data: str):
        interpreter = ModelInterpreter(self.model, data, target_column='text', model_type='text')
        return interpreter.explain_prediction(data)
    
    def train_with_unlabeled_data(self, labeled_data, unlabeled_data):
        trainer = AdvancedSemiSupervisedTrainer(self.model, labeled_data, unlabeled_data, self.config.get('device'), self.config.get('num_classes'))
        trainer.train(epochs=10)
    
    def setup_rl_agent(self, state_size, action_size):
        self.rl_agent = DQNAgent(state_size, action_size)

    def rl_decide(self, state):
        return self.rl_agent.act(state)
    
    def explain_code_result(self, result: str) -> str:
        prompt = f"请解释以下Python代码执行结果：\n{result}\n解释："
        return self.assistant.generate_response(prompt)

    # 新增方法
    def extract_keywords(self, text: str) -> List[str]:
        return self.assistant.extract_keywords(text)

    def plan_task(self, task_description: str) -> str:
        language = self.assistant.detect_language(task_description)
        return self.assistant.plan_task(task_description, language)

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        return self.assistant.translate(text, source_lang, target_lang)

    def handle_qa(ai_nirvana):
        question = input("请输入您的问题：")
        answer = ai_nirvana.assistant.answer_question(question)
        if answer:
            print(f"回答：{answer}")
        else:
            print("很抱歉，我无法回答这个问题。")
        return {"continue": True}
    
    def process_command(self, command: str, *args):
        try:
            print("正在处理，请稍候...")
            
            timeout = 60  # 60秒
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(getattr(self, command), *args)
                try:
                    result = future.result(timeout=timeout)
                    return result
                except concurrent.futures.TimeoutError:
                    print("操作超时，请稍后再试。")
                    return None
        except Exception as e:
            logger.error(f"处理命令时出错: {str(e)}")
            print(f"发生错误: {str(e)}")
            return None

    def initialize_active_learner(self, X_pool, y_pool, X_test, y_test):
        self.active_learner = ActiveLearner(X_pool, y_pool, X_test, y_test, random_state=42)

    def perform_active_learning(self, initial_samples, n_iterations, samples_per_iteration, strategy='uncertainty'):
        if self.active_learner is None:
            raise ValueError("Active learner not initialized. Call initialize_active_learner first.")
    
        print("Starting active learning loop. Press Ctrl+C to interrupt.")
        accuracy_history = []
        try:
            with tqdm(total=n_iterations, desc="Active Learning Progress") as pbar:
                for i in range(n_iterations):
                    start_time = time.time()
                    if i == 0:
                        # 处理初始样本
                        self.active_learner.label_samples(range(initial_samples))
                        accuracy = self.active_learner.active_learning_step(0, strategy)  # 0 表示不需要额外采样
                    else:
                        accuracy = self.active_learner.active_learning_step(samples_per_iteration, strategy)
                    accuracy_history.append(accuracy)
                    elapsed_time = time.time() - start_time
                    pbar.set_postfix({"Accuracy": f"{accuracy:.4f}", "Time": f"{elapsed_time:.2f}s"})
                    pbar.update(1)
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nActive learning interrupted by user.")
    
        final_accuracy = accuracy_history[-1] if accuracy_history else None
        self.accuracy_history = accuracy_history
        return final_accuracy, accuracy_history
    
    def get_active_learning_model(self):
        if self.active_learner is None:
            raise ValueError("Active learner not initialized.")
        return self.active_learner.get_model()

    def set_active_learning_model(self, new_model):
        if self.active_learner is None:
            raise ValueError("Active learner not initialized.")
        self.active_learner.set_model(new_model)

    def create_active_learning_committee(self, n_models=3):
        if self.active_learner is None:
            raise ValueError("Active learner not initialized.")
        self.active_learner.create_committee(n_models)

    def plot_active_learning_curve(self, accuracy_history):
        if self.active_learner is None:
            raise ValueError("Active learner not initialized.")
        self.active_learner.plot_learning_curve(accuracy_history)

    def clean_active_learning_data(self):
        if self.active_learner is not None:
            del self.active_learner.X_pool
            del self.active_learner.y_pool
            del self.active_learner.X_test
            del self.active_learner.y_test
            import gc
            gc.collect()
            print("Active learning data cleaned to free up memory.")    
    
    def load_data_for_active_learning(self):
        self.X_pool, self.y_pool, self.X_test, self.y_test = load_data_for_active_learning()
        print("数据已加载到 AINirvana 实例中。")

    def initialize_active_learner(self):
        if not hasattr(self, 'X_pool'):
            print("数据尚未加载。正在加载示例数据...")
            self.load_data_for_active_learning()
        
        self.active_learner = ActiveLearner(self.X_pool, self.y_pool, self.X_test, self.y_test, random_state=42)
        print("主动学习器已初始化。")

    def label_initial_data(self, n_samples=10):
        if self.active_learner is None:
            raise ValueError("Active learner not initialized. Please use 'init_active_learner' command first.")
    
        initial_indices = self.active_learner.uncertainty_sampling(n_samples)
        self.active_learner.label_samples(initial_indices)
        print(f"{n_samples} samples have been labeled.")
    
    def set_active_learning_model(self, new_model):
        if self.active_learner is None:
            raise ValueError("Active learner not initialized. Please use 'init_active_learner' command first.")
        self.active_learner.set_model(new_model)
        print(f"Model updated to: {type(new_model).__name__}")
   
    def view_committee(self):
        if hasattr(self.active_learner, 'committee'):
            print(f"Active learning committee consists of {len(self.active_learner.committee)} models:")
            for i, model in enumerate(self.active_learner.committee, 1):
                print(f"  Model {i}: {type(model).__name__}")
        else:
            print("No active learning committee has been created yet.")
    def set_model(self, new_model):
        self.model = new_model
        self.is_fitted = False  # 重置拟合状态，因为这是一个新模型
    
    def is_active_learner_initialized(self):
        return self.active_learner is not None
    
    def initialize_feature_engineer(self, data, target_column):
        self.feature_engineer = AutoFeatureEngineer(data, target_column)
        print("Feature engineer initialized successfully.")

    def create_entity_set(self, index_column, time_index=None):
        if index_column not in self.data.columns:
            print(f"索引列 '{index_column}' 不存在，正在创建...")
            self.data[index_column] = range(len(self.data))
    
        self.entityset = ft.EntitySet(id="data")
        self.entityset = self.entityset.add_dataframe(
            dataframe_name="data",
            dataframe=self.data,
            index=index_column,
            time_index=time_index
        )
        print("实体集创建成功。")
        return self.entityset

    def generate_features(self, max_depth, primitives=None):
        if self.feature_engineer is None:
            raise ValueError("Feature engineer not initialized. Call initialize_feature_engineer first.")
    
        if primitives is None:
            # 使用默认原语
            primitives = ["count", "sum", "mean", "max", "min", "std"]  # 示例默认原语
    
        return self.feature_engineer.generate_features(max_depth, primitives)

    def get_important_features(self, n=10, method='correlation'):
        if self.feature_engineer is None:
            raise ValueError("Feature engineer not initialized. Call initialize_feature_engineer first.")
        return self.feature_engineer.get_important_features(n, method)

    def remove_low_information_features(self, threshold=0.95):
        if self.feature_engineer is None:
            raise ValueError("Feature engineer not initialized. Call initialize_feature_engineer first.")
        return self.feature_engineer.remove_low_information_features(threshold)

    def remove_highly_correlated_features(self, threshold=0.9):
        if self.feature_engineer is None:
            raise ValueError("Feature engineer not initialized. Call initialize_feature_engineer first.")
        return self.feature_engineer.remove_highly_correlated_features(threshold)

    def create_custom_feature(self, feature_name, function):
        if self.feature_engineer is None:
            raise ValueError("Feature engineer not initialized. Call initialize_feature_engineer first.")
        self.feature_engineer.create_custom_feature(feature_name, function)

    def get_feature_types(self):
        if self.feature_engineer is None:
            raise ValueError("Feature engineer not initialized. Call initialize_feature_engineer first.")
        return self.feature_engineer.get_feature_types()

    def get_feature_descriptions(self):
        if self.feature_engineer is None:
            raise ValueError("Feature engineer not initialized. Call initialize_feature_engineer first.")
        return self.feature_engineer.get_feature_descriptions()

    def get_feature_matrix(self):
        if self.feature_engineer is None:
            raise ValueError("Feature engineer not initialized. Call initialize_feature_engineer first.")
        return self.feature_engineer.get_feature_matrix()

    def normalize_features(self, method='standard'):
        if self.feature_engineer is None:
            raise ValueError("Feature engineer not initialized. Call initialize_feature_engineer first.")
        self.feature_engineer.normalize_features(method)

    def encode_categorical_features(self, method='onehot'):
        if self.feature_engineer is None:
            raise ValueError("Feature engineer not initialized. Call initialize_feature_engineer first.")
        self.feature_engineer.encode_categorical_features(method)    

    def create_entity_set(self, index_column, time_index=None):
        if self.feature_engineer is None:
            raise ValueError("特征工程器尚未初始化。请先使用 'init_feature_engineer' 命令。")
        return self.feature_engineer.create_entity_set(index_column, time_index)    
    
    def initialize_modules(self):
        try:
            # 使用自定义模型创建 DigitalTwin 实例
            self.create_custom_digital_twin()
            print("数字孪生系统已自动初始化。")

            self.lang_graph = LangGraph()
            print("LangGraph 模块已初始化。")
        except Exception as e:
            print(f"数字孪生系统或 LangGraph 模块初始化失败: {e}")

    def create_custom_digital_twin(self):
        # 定义模型函数
        model = lambda state, t: [state[0] + 1, state[1] * 0.99]

        def detect_anomalies(sensor_data):
            clf = IsolationForest(contamination=0.1, random_state=42)
            anomalies = clf.fit_predict(sensor_data.reshape(-1, 1))
            return np.where(anomalies == -1)[0]  # 返回异常数据的索引

        # 定义 optimize 方法
        def optimize(objective_function, constraints):
            from scipy.optimize import minimize
            import numpy as np

            # 初始猜测值
            initial_guess = [1.0, 500.0]  # 或者其他接近预期解的猜测值

            # 转换约束条件
            constraint_dicts = [{'type': c_type, 'fun': c_func} for c_func, c_type in constraints]

            # 设置优化选项，增加最大迭代次数和函数评估次数
            options = {'maxiter': 10000, 'maxfev': 50000, 'disp': True, 'rhobeg': 1.0, 'tol': 1e-6}

            # 使用 'COBYLA' 算法
            result = minimize(objective_function, initial_guess, constraints=constraint_dicts, method='COBYLA', options=options)

            if not result.success:
                print(f"Optimization warning: {result.message}")
                print(f"Final optimization result: {result.x}")
                print(f"Final objective value: {result.fun}")
    
            return result.x  # 返回优化后的参数，即使优化不成功
       
        # 将方法添加到模型
        model.detect_anomalies = detect_anomalies
        model.optimize = optimize

        # 创建 DigitalTwin 实例
        self.digital_twin = DigitalTwin(model)
        print("数字孪生系统已创建。")
        
    def get_physical_system_model(self):
        # 这是物理系统模型函数的占位符，可以根据需求调整
        def model(state, t):
            # 定义物理系统的状态方程
            return state  # 这是一个简单的占位符，需要根据实际情况定义
        return model
    
    def create_digital_twin(self, model_func):
        # 定义异常检测方法
        def detect_anomalies(sensor_data):
            anomalies = [x for x in sensor_data if x > 500]
            return anomalies

        # 将 detect_anomalies 方法附加到模型函数
        model_func.detect_anomalies = detect_anomalies

        # 定义优化方法
        def optimize(objective_function, constraints):
            from scipy.optimize import minimize

            # 初始猜测
            initial_guess = [1, 1]  # 根据实际情况调整

            # 转换约束条件
            constraint_dicts = [{'type': c_type, 'fun': c_func} for c_func, c_type in constraints]

            # 执行优化
            result = minimize(objective_function, initial_guess, constraints=constraint_dicts)

            if not result.success:
                raise ValueError("Optimization failed: " + result.message)
        
            return result.x  # 返回优化后的参数

        # 将 optimize 方法附加到模型函数
        model_func.optimize = optimize

        # 初始化数字孪生系统
        self.digital_twin = DigitalTwin(model_func)
        print("数字孪生系统已创建并附加了异常检测和优化方法。")

    def simulate_digital_twin(self, initial_conditions, time_steps):
        if self.digital_twin:
            return self.digital_twin.simulate(initial_conditions, time_steps)
        else:
            print("数字孪生系统尚未初始化。")

    def monitor_digital_twin(self, sensor_data):
        if self.digital_twin:
            return self.digital_twin.monitor(sensor_data)
        else:
            print("数字孪生系统尚未初始化。")

    def optimize_digital_twin(self, objective_function, constraints) -> Dict:
        if self.digital_twin is None:
            return {"error": "Digital twin has not been created yet."}
        try:
            optimal_params = self.digital_twin.optimize(objective_function, constraints)
            return {"success": True, "optimal_parameters": optimal_params.tolist()}
        except Exception as e:
            logger.error(f"Error in digital twin optimization: {str(e)}")
            return {"error": str(e)}

    def update_digital_twin_model(self, new_model_func):
        if self.digital_twin is None:
            print("Error: Digital twin has not been created yet. Use 'create_digital_twin' first.")
            return
    
        # 定义异常检测方法
        def detect_anomalies(sensor_data):
            anomalies = [x for x in sensor_data if x > 500]
            return anomalies

        # 将 detect_anomalies 方法附加到新的模型函数
        new_model_func.detect_anomalies = detect_anomalies

        # 定义优化方法
        def optimize(objective_function, constraints):
            return self.digital_twin.optimize(objective_function, constraints)

        # 将 optimize 方法附加到模型函数
        new_model_func.optimize = optimize

        # 更新数字孪生模型
        self.digital_twin.update_model(new_model_func)
        print("物理系统模型已更新并附加了异常检测和优化方法。")

    def validate_digital_twin_model(self, validation_data):
        if self.digital_twin is None:
            print("Error: Digital twin has not been created yet. Use 'create_digital_twin' first.")
            return
    
        # 确保 validation_data 是 numpy 数组
        if isinstance(validation_data, str):
            validation_data = np.array([float(x) for x in validation_data.split()])
        elif not isinstance(validation_data, np.ndarray):
            validation_data = np.array(validation_data)
    
        # 将验证数据分为初始条件和时间步长
        initial_condition = validation_data[0]
        time_steps = np.arange(len(validation_data))
    
        # 使用模型进行预测
        predicted_values = self.digital_twin.simulate([initial_condition], time_steps)
    
        # 计算均方误差
        mse = np.mean((predicted_values[:, 0] - validation_data) ** 2)
    
        # 将均方误差转换为准确度分数
        accuracy = 1 / (1 + mse)
    
        print(f"模型验证准确性: {accuracy}")
        print(f"预测值: {predicted_values[:, 0]}")
        print(f"实际值: {validation_data}")
        return accuracy
   
    def optimize_digital_twin(self, objective_function, constraints) -> Dict:
        if self.digital_twin is None:
            return {"error": "Digital twin has not been created yet."}
        try:
            optimal_params = self.digital_twin.optimize(objective_function, constraints)
            return {"success": True, "optimal_parameters": optimal_params.tolist()}
        except Exception as e:
            logger.error(f"Error in digital twin optimization: {str(e)}")
            return {"error": str(e)} 

    def generate_text(self, prompt, max_tokens=100, temperature=0.7):
        result = self.generative_ai.generate_text(prompt, max_tokens=max_tokens, temperature=temperature)
        if result:
            print(f"生成的文本（检测到的语言：{detect(prompt)}）：")
            print(result)
        else:
            print("生成文本时出错。")
        return result

    def classify_image(self, image_path):
        return self.generative_ai.classify_image(image_path)

    def generate_image_caption(self, image_path):
        return self.generative_ai.generate_image_caption(image_path)

    def fine_tune(self, train_data, epochs=1, learning_rate=2e-5, batch_size=2):
        return self.generative_ai.fine_tune(train_data, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)

    def save_model(self, filename: str, model_type: str = 'general'):  
        try:  
            if model_type == 'general':  
                # 保存通用模型（如果有的话）  
                self.generative_ai.save_model(filename)  
            elif model_type == 'agent':  
                # 保存智能代理模型  
                if hasattr(self, 'agent'):  
                    self.agent.save(filename)  
                    # 假设 agent 有 state_size 和 action_size 属性  
                    np.savez(filename + '_params.npz', state_size=self.agent.state_size, action_size=self.agent.action_size)  
                else:  
                    raise ValueError("No agent model to save.")  
            else:  
                raise ValueError("Invalid model type. Use 'general' or 'agent'.")  
            print(f"Saved {model_type} model to {filename}")  
            return {"message": f"{model_type.capitalize()} model saved successfully.", "continue": True}  
        except Exception as e:  
            return {"error": str(e), "continue": True}  

    def load_model(self, filename: str, model_type: str = 'general'):  
        try:  
            if model_type == 'general':  
                # 加载通用模型  
                self.generative_ai.load_model(filename)  
            elif model_type == 'agent':  
                # 加载智能代理模型  
                if not hasattr(self, 'agent'):  
                    # 如果 agent 还没有被初始化，我们需要先创建它  
                    params = np.load(filename + '_params.npz')  
                    state_size = params['state_size']  
                    action_size = params['action_size']  
                    self.agent = DQNAgent(state_size, action_size)  
                self.agent.load(filename)  
            else:  
                raise ValueError("Invalid model type. Use 'general' or 'agent'.")  
            print(f"Loaded {model_type} model from {filename}")  
            return {"message": f"{model_type.capitalize()} model loaded successfully.", "continue": True}  
        except Exception as e:  
            return {"error": str(e), "continue": True}
        
    def create_agent(self, state_size: int, action_size: int, verbose=True):  
        self.agent = DQNAgent(state_size, action_size, verbose=verbose)  
        print(f"DQN agent created with state size {state_size} and action size {action_size}")

    def train_agent(self, environment_name: str, episodes: int, max_steps: int):  
        import gym  
        import numpy as np  

        env = None  
        try:  
            environment_name = environment_name.strip()  
            print(f"Creating environment: '{environment_name}'")  
            env = gym.make(environment_name)  

            if not hasattr(self, 'agent'):  
                state_size = env.observation_space.shape[0]  
                action_size = env.action_space.n  
                print(f"State size: {state_size}, Action size: {action_size}")  
                self.agent = DQNAgent(state_size, action_size)  

            print("Checking DQNAgent attributes before training:")  
            self.check_agent_attributes()  

            for episode in range(episodes):  
                state = env.reset()  
                if isinstance(state, tuple):  
                    state = state[0]  
                print(f"Initial state shape: {np.array(state).shape}")  
                state = self.agent.process_state(state)  
                total_reward = 0  
                for step in range(max_steps):  
                    action = self.agent.act(state, train=True)  
                    step_result = env.step(action)  
                
                    if len(step_result) == 5:  
                        next_state, reward, terminated, truncated, _ = step_result  
                        done = terminated or truncated  
                    else:  
                        next_state, reward, done, _ = step_result  
                
                    print(f"Next state shape before processing: {np.array(next_state).shape}")  
                    next_state = self.agent.process_state(next_state)  
                    self.agent.remember(state, action, reward, next_state, done)  
                    state = next_state  
                    total_reward += reward  

                    if len(self.agent.memory) > self.agent.batch_size:  
                        self.agent.replay(self.agent.batch_size)  
                
                    if done:  
                        break  

                print(f"Episode: {episode+1}/{episodes}, Score: {total_reward}, Epsilon: {self.agent.epsilon:.4f}")  
                print(f"Train count: {self.agent.train_count}")  

                self.agent.epsilon = max(self.agent.epsilon_min, self.agent.epsilon * self.agent.epsilon_decay)  

                if episode % self.agent.update_target_frequency == 0:  
                    self.agent.update_target_model()  

            print(f"Trained agent for {episodes} episodes in {environment_name}")  
        except Exception as e:  
            print(f"An error occurred during training: {e}")  
            import traceback  
            traceback.print_exc()  
        finally:  
            if env:  
                env.close()

    def run_agent(self, environment_name: str, episodes: int, max_steps: int):  
        import gym  
        import numpy as np  

        try:  
            env = gym.make(environment_name)  
            if not hasattr(self, 'agent'):  
                state_size = env.observation_space.shape[0]  
                action_size = env.action_space.n  
                self.agent = DQNAgent(state_size, action_size)  
                print("Warning: No trained agent found. Using a new, untrained agent.")  
            else:  
                print("Using existing agent.")  

            total_rewards = []  

            for episode in range(episodes):  
                state = env.reset()  
                if isinstance(state, tuple):  
                    state = state[0]  # 如果 reset() 返回元组，取第一个元素  
            
                state = self.agent.process_state(state)  
                episode_reward = 0  

                for step in range(max_steps):  
                    action = self.agent.act(state, train=False)  # 使用贪婪策略  
                
                    step_result = env.step(action)  
                    if len(step_result) == 4:  
                        next_state, reward, done, _ = step_result  
                    elif len(step_result) == 5:  
                        next_state, reward, terminated, truncated, _ = step_result  
                        done = terminated or truncated  
                    else:  
                        raise ValueError(f"Unexpected number of return values from env.step(): {len(step_result)}")  
                
                    next_state = self.agent.process_state(next_state)  
                    episode_reward += reward  
                
                    state = next_state  
                
                    if done:  
                        break  

                total_rewards.append(episode_reward)  
                print(f"Episode: {episode+1}/{episodes}, Score: {episode_reward}")  

            average_reward = np.mean(total_rewards)  
            print(f"\nAverage score over {episodes} episodes: {average_reward}")  
        
        except Exception as e:  
            print(f"An error occurred while running the agent: {e}")  
            import traceback  
            traceback.print_exc()  

    def setup_rl_agent(self, state_size, action_size):
        self.rl_agent = DQNAgent(state_size, action_size)
        print(f"RL agent set up with state size {state_size} and action size {action_size}")

    def rl_decide(self):  
        try:  
            state_input = input("Enter the current state (comma-separated values): ")  
            state = np.array([float(x.strip()) for x in state_input.split(',')])  
        
            # 确保状态是二维数组，形状为 (1, state_size)  
            state = state.reshape(1, -1)  
        
            if not hasattr(self, 'rl_agent'):  
                return {"error": "RL agent not initialized. Please use 'setup_rl_agent' first.", "continue": True}  
        
            action = self.rl_agent.act(state)  
        
            print(f"Chosen action: {action}")  
            return {"message": f"RL decision made. Chosen action: {action}", "continue": True}  
        except ValueError as ve:  
            return {"error": f"Invalid input: {str(ve)}", "continue": True}  
        except Exception as e:  
            return {"error": f"Error in RL decision: {str(e)}", "continue": True}      

    def check_agent_attributes(self):
        if hasattr(self, 'agent'):
            print("DQNAgent attributes:")
            for attr in dir(self.agent):
                if not attr.startswith('__'):
                    print(f"- {attr}")
        else:
            print("Agent has not been created yet.") 

    def add_entity(self, entity: str, entity_type: str):  
        return self.lang_graph.add_entity(entity, entity_type)  

    def update_entity(self, entity: str, new_properties: dict):  
        return self.lang_graph.update_entity(entity, new_properties)  

    def delete_entity(self, entity: str):  
        return self.lang_graph.delete_entity(entity) 
    
    def add_relation(self, entity1: str, entity2: str, relation: str):
        self.lang_graph.add_relationship(entity1, entity2, relation)
        print(f"关系 {relation} 已添加，实体 {entity1} 和 {entity2} 之间。")        
    
    def get_graph_summary(self):  
        return self.lang_graph.get_graph_summary()
    
    def export_graph(self, format):  
        if format.lower() not in ['graphml', 'gexf']:  
            return "不支持的格式。请选择 graphml 或 gexf。"  
        
        filename = f"graph.{format.lower()}"  
        if format.lower() == 'graphml':  
            nx.write_graphml(self.lang_graph.graph.get_networkx_graph(), filename)  
        else:  # gexf  
            nx.write_gexf(self.lang_graph.graph.get_networkx_graph(), filename)  
        
        return f"Graph exported as {filename}"  

    def infer_commonsense(self, context):  
        try:  
            response = self.api_client.chat_completion([  
                {"role": "system", "content": "You are a helpful assistant that provides commonsense inferences."},  
                {"role": "user", "content": f"Given the context: '{context}', provide a commonsense inference."}  
            ])  
            if response and 'choices' in response and len(response['choices']) > 0:  
                return response['choices'][0]['message']['content'].strip()  
            else:  
                logging.error("Unexpected response format from API")  
                return None  
        except Exception as e:  
            logging.error(f"Error in infer_commonsense: {str(e)}")  
            return None  
    
    def retrieve_knowledge(self, query: str) -> Dict[str, Any]:
        try:
            response = self.lang_graph.retrieve_knowledge(query)
            
            if 'error' in response:
                logger.warning(f"知识检索返回错误: {response['error']}")
                print(f"知识检索出错: {response['error']}")
                return response

            logger.info(f"成功处理知识检索: 查询='{query}'")
            print(f"查询: {response['query']}")
            print(f"结果: {response['result']}")
            
            if response['source_documents']:
                print("\n相关文档:")
                for i, doc in enumerate(response['source_documents'], 1):
                    print(f"{i}. {doc['content']}")
                    if 'source' in doc['metadata']:
                        print(f"   来源: {doc['metadata']['source']}")
            
            return response
        except Exception as e:
            logger.error(f"知识检索处理出错: {str(e)}")
            return {"error": "知识检索处理失败", "continue": True}

    def interactive_knowledge_retrieval(self):
        while True:
            query = input("\n请输入您的问题（输入'quit'退出）: ")
            if query.lower() == 'quit':
                break
            
            response = self.retrieve_knowledge(query)
            
            if 'error' not in response:
                follow_up = input("\n您还有其他问题吗？(yes/no): ")
                if follow_up.lower() != 'yes':
                    break
    
    def semantic_search(self, query: str, k: int = 5):
        results = self.lang_graph.semantic_search(query, k)
        print(f"语义搜索结果: {results}")  

    def add_relationship(self, entity1: str, entity2: str, relationship: str) -> None:  
        self.lang_graph.add_relationship(entity1, entity2, relationship)               

    def get_related_entities(self, entity):
        try:
            related = self.lang_graph.get_related_entities(entity)
            print(f"与 '{entity}' 相关的实体: {related}")
            return related
        except Exception as e:
            print(f"Error in get_related_entities: {str(e)}")
            return []

    def get_all_entities(self):
        try:
            entities = self.lang_graph.get_all_entities()
            print(f"所有实体: {entities}")
            return entities
        except Exception as e:
            print(f"Error in get_all_entities: {str(e)}")
            return []

    def get_entity_info(self, entity):
        try:
            info = self.lang_graph.get_entity_info(entity)
            print(f"实体 '{entity}' 的信息: {info}")
            return {"result": info, "continue": True}
        except Exception as e:
            print(f"Error in get_entity_info: {str(e)}")
            return {"result": {}, "continue": True}

    def get_all_relationships(self):
        try:
            relationships = self.lang_graph.get_all_relationships()
            print(f"所有关系: {relationships}")
            return relationships
        except Exception as e:
            print(f"Error in get_all_relationships: {str(e)}")
            return []    

    def run_agent(self, query):
        try:
            response = self.lang_graph.run_agent(query)
            print(f"Agent 响应: {response}")
            return response
        except Exception as e:
            print(f"Error in run_agent: {str(e)}")
            return "An error occurred while running the agent."    
    
def load_data_for_active_learning():
    # 这里我们使用一个简单的合成数据集作为示例
    # 在实际应用中，您可能需要从文件或数据库加载真实数据
    print("正在生成示例数据集...")
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=2,
                               n_repeated=0, n_classes=2, n_clusters_per_class=2, random_state=42)
    
    # 将数据分为训练集（作为未标记池）和测试集
    X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"生成的数据集大小：")
    print(f"未标记池：{X_pool.shape[0]} 样本")
    print(f"测试集：{X_test.shape[0]} 样本")
    
    return X_pool, y_pool, X_test, y_test

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def get_new_model_from_user():
    print("Select a new model:")
    print("1. Random Forest")
    print("2. Support Vector Machine")
    print("3. Logistic Regression")
    choice = input("Enter your choice (1/2/3): ")
    
    if choice == '1':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=42)
    elif choice == '2':
        from sklearn.svm import SVC
        return SVC(random_state=42)
    elif choice == '3':
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(random_state=42)
    else:
        print("Invalid choice. Using default Random Forest.")
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=42)
    
def handle_sentiment(ai_nirvana):
    text = input("请输入要分析情感的文本：")
    sentiment = ai_nirvana.analyze_sentiment(text)
    if "error" in sentiment:
        print(f"情感分析出错: {sentiment['error']}")
        print("请检查日志以获取更多详细信息。")
    else:
        print("情感分析结果:")
        for key, value in sentiment.items():
            print(f"  {key}: {value:.2f}")
    return {"continue": True}

def handle_command(command: str, ai_nirvana: AINirvana) -> Dict[str, Any]:  
    command = command.lower().strip()
    try:
        command = command.lower().strip()  
        logger.info(f"Handling command: {command}") 
        
        if command not in AVAILABLE_COMMANDS:
            response = ai_nirvana.process(command)
            print_user_input(command)
            print("\n回答：")
            print_assistant_response(response)
            print_dialogue_context(ai_nirvana.dialogue_manager.get_dialogue_context())
            return {"continue": True}

        if command == "clear":
            ai_nirvana.dialogue_manager.clear_history()
            ai_nirvana.assistant.clear_context()
            ai_nirvana.variable_state.clear()
            return {"message": "对话历史和变量状态已清除。", "continue": True}
        elif command in ["help", "get help", "get_help"]:  
            help_message = get_help()  
            print(help_message)  
            return {"message": "请按照以上帮助信息操作。", "continue": True}  
        elif command == "quit":
            return {"message": "谢谢使用 AI Nirvana 智能助手，再见！", "continue": False}
        elif command == "sentiment":
            text = input("请输入要分析情感的文本：")
            sentiment = ai_nirvana.assistant.analyze_sentiment(text)
            print(f"Sentiment: {sentiment}")
            return {"continue": True}
        elif command == "summarize":
            print("请输入要生成摘要的文本（输入空行结束）：")
            lines = []
            while True:
                line = input()
                if line.strip() == "":
                    break
                lines.append(line)
            text = "\n".join(lines)
            summary = ai_nirvana.summarize(text)  # 只传递一个参数
            print(f"摘要：{summary}")
            return {"continue": True}
        elif command == "change_model":
            available_models = ai_nirvana.get_available_models()
            print(f"可用的模型有：{', '.join(available_models)}")
            model_name = input("请输入新的模型名称（或输入 'cancel' 取消）：").strip()
            if model_name == "cancel":
                return {"message": "已取消更改模型。", "continue": True}
            if model_name in available_models:
                ai_nirvana.change_model(model_name)
                return {"message": f"模型已更改为 {model_name}。", "continue": True}
            else:
                print(f"无效的模型名称：{model_name}")
                return {"continue": True}
        elif command == "execute":
            print("请输入要执行的 Python 代码（输入空行结束）：")
            code_lines = []
            while True:
                line = input()
                if line.strip() == "":
                    break
                code_lines.append(line)
            code = "\n".join(code_lines)
            result, local_vars = ai_nirvana.execute_code(code)  # 注意这里的改变
            ai_nirvana.update_variable_state(local_vars)  # 更新变量状态
            print("\n" + "="*40)
            print("执行结果:")
            print(result)
            print("="*40 + "\n")
            explanation = input("需要进一步解释结果吗？(yes/no): ")
            if explanation.lower() == 'yes':
                explanation_result = ai_nirvana.explain_code_result(code, result)
                print("解释：", explanation_result)
            return {"continue": True}

        elif command == "validate_code":
            print("请输入要验证的 Python 代码（输入空行结束）：")
            code_lines = []
            while True:
                line = input()
                if line.strip() == "":
                    break
                code_lines.append(line)
            code = "\n".join(code_lines)
            is_valid = ai_nirvana.validate_code(code)
            if is_valid:
                print("代码验证通过，是安全的。")
            else:
                print("代码验证失败，可能存在安全风险。")
            return {"continue": True}
        elif command == "supported_languages":
            languages = ai_nirvana.get_supported_languages()
            print("支持的编程语言:")
            for lang in languages:
                print(f"- {lang}")
            return {"continue": True}
        elif command == "vars":
            print("当前定义的变量:")
            print(ai_nirvana.get_variable_state())
            return {"continue": True}
        elif command == "explain":
            data = input("请输入要解释的数据：")
            explanation = ai_nirvana.explain_model(data)
            print(f"模型解释：{explanation}")
            return {"continue": True}
        elif command == "encrypt":
            data = input("请输入要加密的数据：")
            try:
                encrypted = ai_nirvana.encrypt_sensitive_data(data)
                print(encrypted)
            except AIAssistantException as e:
                print(f"加密失败: {str(e)}")
            return {"continue": True}
        elif command == "decrypt":
            encrypted_data = input("请输入要解密的数据：")
            result = ai_nirvana.decrypt_sensitive_data(encrypted_data)
            print(result)
            return {"continue": True}
        elif command == "add_knowledge":
            key = input("请输入知识的键：")
            value = input("请输入知识的值：")
            ai_nirvana.add_knowledge(key, value)
            return {"message": f"知识已添加: {key}", "continue": True}

        elif command == "get_knowledge":
            key = input("请输入要检索的知识键：")
            try:
                value = ai_nirvana.get_knowledge(key)
                print(f"知识内容：{value}")
            except KeyError as e:
                print(f"错误：{str(e)}")
            return {"continue": True}

        elif command == "list_knowledge":
            all_knowledge = ai_nirvana.list_all_knowledge()
            print("所有知识：")
            for k, v in all_knowledge.items():
                print(f"{k}: {v}")
            return {"continue": True}

        elif command == "update_knowledge":
            key = input("请输入要更新的知识键：")
            value = input("请输入新的知识值：")
            try:
                ai_nirvana.knowledge_base.update_knowledge(key, value)
                print(f"知识已更新: {key}")
            except KeyError as e:
                print(f"错误：{str(e)}")
            return {"continue": True}

        elif command == "delete_knowledge":
            key = input("请输入要删除的知识键：")
            try:
                ai_nirvana.knowledge_base.delete_knowledge(key)
                print(f"知识已删除: {key}")
            except KeyError as e:
                print(f"错误：{str(e)}")
            return {"continue": True}

        elif command == "search_knowledge":
            query = input("请输入搜索关键词：").strip()
            results = ai_nirvana.search_knowledge(query)
            if results:
                print("搜索结果：")
                for k, v in results.items():
                    print(f"{k}: {v}")
            else:
                 print("未找到相关知识。")
            return {"continue": True}

        elif command == "extract_keywords":
            text = input("请输入要提取关键词的文本：")
            keywords = ai_nirvana.assistant.extract_keywords(text)
            print(f"提取的关键词：{', '.join(keywords)}")
            return {"continue": True}
        elif command == "plan_task":
            task = input("请输入要规划的任务描述：")
            plan = ai_nirvana.assistant.plan_task(task)
            print(f"任务计划：\n{plan}")
            return {"continue": True}
        elif command == "translate":
            print("请输入要翻译的文本（输入空行结束）：")
            lines = []
            while True:
                line = input()
                if line.strip() == "":
                    break
                lines.append(line)
            text = "\n".join(lines)
            target_lang = input("请输入目标语言（如 'en' 表示英语，'zh' 表示中文）：")
            translated = ai_nirvana.assistant.translate(text, target_lang)
            print(f"翻译结果：{translated}")
            return {"continue": True}
        elif command == "test_complex":
            test_complex_text_analysis(ai_nirvana)
            return {"continue": True}
        elif command == "test_models":
            test_model_comparison(ai_nirvana)
            return {"continue": True}
        elif command == "test_code_safety":  
            print("开始执行代码安全性测试...")  
            try:  
                from tests.code_execution_safety import test_code_execution_safety  
                test_result = test_code_execution_safety(ai_nirvana)  
                print(f"测试结果: {test_result}")  
            except Exception as e:  
                print(f"执行测试时发生错误: {str(e)}")  
                print("错误详情:")  
                import traceback  
                traceback.print_exc()  
            print("代码安全性测试完成。")  
            print("正在返回主循环...")  
            return {"continue": True}
        elif command == "qa":
            question = input("请输入您的问题：")
            answer = ai_nirvana.assistant.answer_question(question)
            print(f"回答：{answer}")
            return {"continue": True}
        elif command == "init_active_learner":
            ai_nirvana.initialize_active_learner()
            return {"continue": True}
        elif command == "run_active_learning":
            if ai_nirvana.active_learner is None:
                print("Active learner not initialized. Please use 'init_active_learner' command first.")
                return {"continue": True}
            initial_samples = int(input("Enter initial samples: "))
            n_iterations = int(input("Enter number of iterations: "))
            samples_per_iteration = int(input("Enter samples per iteration: "))
            strategy = input("Enter sampling strategy (default: uncertainty): ") or "uncertainty"
            final_accuracy, accuracy_history = ai_nirvana.perform_active_learning(
                initial_samples, n_iterations, samples_per_iteration, strategy
            )
            if final_accuracy is not None:
                print(f"Final accuracy: {final_accuracy}")
            choice = input("Do you want to plot the learning curve? (y/n): ")
            if choice.lower() == 'y':
                ai_nirvana.plot_active_learning_curve(accuracy_history)
            choice = input("Do you want to clean the active learning data to free up memory? (y/n): ")
            if choice.lower() == 'y':
                ai_nirvana.clean_active_learning_data()
            return {"continue": True}
        elif command == "al_model":
            while True:
                action = input("Enter 'get' to get current model or 'set' to set a new model (or 'cancel' to exit): ").strip().lower()
                if action == 'cancel':
                    print("Operation cancelled.")
                    break
                elif action in ['get', 'set']:
                    if action == "get":
                        model = ai_nirvana.get_active_learning_model()
                        print(f"Current active learning model: {model}")
                    elif action == "set":
                        new_model = get_new_model_from_user()
                        ai_nirvana.set_active_learning_model(new_model)
                        print("New model set for active learning.")
                    break
                else:
                    print("Invalid input. Please enter 'get', 'set', or 'cancel'.")
            return {"continue": True}
        elif command == "al_committee":
            if not ai_nirvana.is_active_learner_initialized():
                print("Error: Active learner not initialized. Please use 'init_active_learner' command first.")
            else:
                try:
                    n_models = int(input("Enter number of models for the committee: "))
                    ai_nirvana.create_active_learning_committee(n_models)
                    print(f"Active learning committee with {n_models} models created.")
                except ValueError as e:
                    print(f"Error: {str(e)}")
            return {"continue": True}
        elif command == "al_plot":
            if hasattr(ai_nirvana, 'accuracy_history'):
                ai_nirvana.plot_active_learning_curve(ai_nirvana.accuracy_history)
                print("Learning curve plotted.")
            else:
                print("No learning history available. Please run active learning first.")
            return {"continue": True}
        elif command == "label_initial_data":
            try:
                n_samples = int(input("Enter the number of initial samples to label: "))
                ai_nirvana.label_initial_data(n_samples)
            except ValueError as e:
                print(f"Error: {str(e)}")
            return {"continue": True}
        elif command == "view_committee":
            ai_nirvana.view_committee()
            return {"continue": True}
        
        elif command == "init_feature_engineer":
            data_source = input("请选择数据源 (1: 示例数据, 2: 本地文件): ")
        
            if data_source == "1":
                 n_samples = int(input("请输入样本数量（默认1000）: ") or "1000")
                 n_features = int(input("请输入特征数量（默认10）: ") or "10")
                 data = AutoFeatureEngineer.generate_sample_data(n_samples, n_features)
                 target_column = 'target'
                 print(f"已生成示例数据，包含 {n_samples} 个样本和 {n_features} 个特征。目标列为 'target'")
            elif data_source == "2":
                data_path = input("请输入数据文件路径: ")
                target_column = input("请输入目标列名称: ")
                try:
                    data = pd.read_csv(data_path)
                    print(f"成功读取数据文件，包含 {data.shape[0]} 个样本和 {data.shape[1]} 个列。")
                except Exception as e:
                    print(f"读取数据文件时出错: {str(e)}")
                    return {"continue": True}
            else:
                print("无效的选择。请输入 1 或 2。")
                return {"continue": True}
        
            try:
                ai_nirvana.initialize_feature_engineer(data, target_column)
                print("特征工程器初始化成功。")
            except Exception as e:
                print(f"初始化特征工程器时出错: {str(e)}")
            return {"continue": True}

        elif command == "create_entity_set":
            index_column = input("请输入索引列名称（如果不存在将自动创建）: ").strip()
            time_index = input("请输入时间索引列名称（如果没有请直接回车）: ").strip() or None
            try:
                ai_nirvana.create_entity_set(index_column, time_index)
                print("实体集创建成功。")
            except Exception as e:
                print(f"创建实体集时出错: {str(e)}")
            return {"continue": True}

        elif command == "generate_features":
            max_depth = int(input("Enter max depth for feature generation: "))
            use_custom = input("Use custom primitives? (y/n): ").lower() == 'y'
            primitives = None
            if use_custom:
                primitives_input = input("Enter primitives (comma-separated) or press Enter for default: ").strip()
                if primitives_input:
                    primitives = [p.strip() for p in primitives_input.split(',')]
                else:
                    print("No custom primitives entered. Using default primitives.")
    
            try:
                feature_matrix, feature_defs = ai_nirvana.generate_features(max_depth, primitives)
                print(f"Generated {len(feature_defs)} features.")
            except Exception as e:
                print(f"Error generating features: {str(e)}")
            return {"continue": True}

        elif command == "get_important_features":
            n = int(input("Enter number of important features to retrieve: "))
            method = input("Enter method (correlation/mutual_info): ")
            important_features = ai_nirvana.get_important_features(n, method)
            print(f"Top {n} important features: {important_features}")
            return {"continue": True}

        elif command == "remove_low_info_features":
            threshold = float(input("Enter threshold for low information features: "))
            removed_features = ai_nirvana.remove_low_information_features(threshold)
            print(f"Removed {len(removed_features)} low information features.")
            return {"continue": True}

        elif command == "remove_correlated_features":
            threshold = float(input("Enter correlation threshold: "))
            removed_features = ai_nirvana.remove_highly_correlated_features(threshold)
            print(f"Removed {len(removed_features)} highly correlated features.")
            return {"continue": True}

        elif command == "create_custom_feature":
            feature_name = input("Enter the name for the custom feature: ")
            function_str = input("Enter the Python function to generate the feature: ")
            function = eval(f"lambda row: {function_str}")
            ai_nirvana.create_custom_feature(feature_name, function)
            print(f"Custom feature '{feature_name}' created successfully.")
            return {"continue": True}

        elif command == "get_feature_types":
            feature_types = ai_nirvana.get_feature_types()
            for feature, type_ in feature_types.items():
                print(f"{feature}: {type_}")
            return {"continue": True}

        elif command == "get_feature_descriptions":
            descriptions = ai_nirvana.get_feature_descriptions()
            for desc in descriptions:
                print(desc)
            return {"continue": True}

        elif command == "normalize_features":
            method = input("Enter normalization method (standard/minmax): ")
            ai_nirvana.normalize_features(method)
            print("Features normalized successfully.")
            return {"continue": True}

        elif command == "encode_categorical_features":
            method = input("Enter encoding method (onehot/label): ")
            ai_nirvana.encode_categorical_features(method)
            print("Categorical features encoded successfully.")
            return {"continue": True}
        
        elif command == "create_digital_twin":
            model_str = input("请输入物理系统模型函数（作为lambda表达式）: ")
            physical_system_model = eval(model_str)
            ai_nirvana.create_digital_twin(physical_system_model)
            return {"continue": True}

        elif command == "simulate_digital_twin":
            initial_conditions = list(map(float, input("请输入初始条件（以空格分隔）: ").split()))
            time_steps = list(map(float, input("请输入时间步长（以空格分隔）: ").split()))
            simulation_result = ai_nirvana.simulate_digital_twin(initial_conditions, time_steps)
            print(f"模拟结果: {simulation_result}")
            return {"continue": True}

        elif command == "monitor_digital_twin":
            sensor_data = np.array(list(map(float, input("请输入传感器数据（以空格分隔）: ").split())))
            anomalies = ai_nirvana.monitor_digital_twin(sensor_data)
            print(f"检测到的异常: {anomalies}")
            return {"continue": True}

        elif command == "optimize_digital_twin":
            if ai_nirvana.digital_twin is None:
                print("Error: Digital twin has not been created yet. Use 'create_digital_twin' first.")
                return {"continue": True}
    
            try:
                objective_function = eval(input("请输入目标函数（作为lambda表达式）: "))
                constraints = eval(input("请输入约束条件（作为列表的元组）: "))
        
                results = ai_nirvana.optimize_digital_twin(objective_function, constraints)
        
                if "error" in results:
                    print(f"Error in optimization: {results['error']}")
                else:
                    print("Optimization completed.")
                    print(f"Optimal parameters: {results['optimal_parameters']}")
            except Exception as e:
                print(f"Error in optimization process: {str(e)}")
    
            return {"continue": True}

        elif command == "update_digital_twin_model":
            new_model = eval(input("请输入新的物理系统模型函数: "))
            ai_nirvana.update_digital_twin_model(new_model)
            print("物理系统模型已更新。")
            return {"continue": True}

        elif command == "validate_digital_twin_model":
            validation_data = np.array(list(map(float, input("请输入验证数据（以空格分隔）: ").split())))
            accuracy = ai_nirvana.validate_digital_twin_model(validation_data)
            print(f"模型验证准确性: {accuracy}")
            return {"continue": True}

        elif command == "generate_text":
            prompt = input("请输入文本生成的提示：")
            max_tokens = int(input("请输入最大生成令牌数（默认100）：") or "100")
            temperature = float(input("请输入温度参数（0-1之间，默认0.7）：") or "0.7")
            ai_nirvana.generate_text(prompt, max_tokens=max_tokens, temperature=temperature)
            return {"continue": True}

        elif command == "classify_image":
            image_path = input("请输入图像路径：").strip()
            # 移除可能的引号
            image_path = image_path.strip("\"'")
            try:
                result = ai_nirvana.classify_image(image_path)
                print("图像分类结果：", result)
            except Exception as e:
                print(f"处理图像时出错: {str(e)}")
            return {"continue": True}

        elif command == "caption_image":
            image_path = input("请输入图像路径：")
            result = ai_nirvana.generative_ai.generate_image_caption(image_path)
            print("图像描述：", result)
            return {"continue": True}

        elif command == "fine_tune_model":
            train_data = input("请输入训练数据（以逗号分隔）：").split(',')
            epochs = int(input("请输入训练轮数（默认1）：") or "1")
            learning_rate = float(input("请输入学习率（默认2e-5）：") or "2e-5")
            batch_size = int(input("请输入批次大小（默认2）：") or "2")
            try:
                ai_nirvana.fine_tune(train_data, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)
                print("模型微调完成")
            except Exception as e:
                print(f"模型微调过程中发生错误: {str(e)}")
            return {"continue": True}

        elif command == "save_model":
            filename = input("Enter filename to save model: ")
            model_type = input("Enter model type (general/agent): ").lower()
            try:
                ai_nirvana.save_model(filename, model_type)
                return {"message": f"{model_type.capitalize()} model saved successfully.", "continue": True}
            except ValueError as e:
                return {"message": f"Error saving model: {str(e)}", "continue": True}

        elif command == "load_model":
            filename = input("Enter filename to load model: ")
            model_type = input("Enter model type (general/agent): ").lower()
            try:
                ai_nirvana.load_model(filename, model_type)
                return {"message": f"{model_type.capitalize()} model loaded successfully.", "continue": True}
            except ValueError as e:
                return {"message": f"Error loading model: {str(e)}", "continue": True}

        elif command == "create_agent":
            state_size = int(input("Enter state size: "))
            action_size = int(input("Enter action size: "))
            ai_nirvana.create_agent(state_size, action_size)
            return {"continue": True}

        elif command == "train_agent":
            environment_name = input("Enter environment name (e.g., 'CartPole-v1'): ")
            episodes = int(input("Enter number of episodes: "))
            max_steps = int(input("Enter maximum steps per episode: "))
            ai_nirvana.train_agent(environment_name, episodes, max_steps)
            return {"continue": True}

        elif command == "run_agent":  
            environment_name = input("请输入环境名称（例如 'CartPole-v1'）：")  
            episodes = int(input("请输入要运行的回合数："))  
            max_steps = int(input("请输入每个回合的最大步数："))  
            ai_nirvana.run_agent(environment_name, episodes, max_steps)  
            return {"continue": True}

        elif command == "setup_rl_agent":
            try:
                state_size = int(input("Enter state size: "))
                action_size = int(input("Enter action size: "))
                ai_nirvana.setup_rl_agent(state_size, action_size)
                return {"message": "RL agent set up successfully.", "continue": True}
            except ValueError as e:
                return {"message": f"Error setting up RL agent: {str(e)}", "continue": True}

        elif command == "rl_decide":  
            result = ai_nirvana.rl_decide()  
            if "error" in result:  
                print(result["error"])  
            else:  
                print(result["message"])  
            return {"continue": True}

        elif command == "add_entity":  
            entity = input("请输入实体名称：")  
            entity_type = input("请输入实体类型：")  
            result = ai_nirvana.add_entity(entity, entity_type)  
            print(f"实体添加结果：{result}")  
            return {"continue": True}

        elif command == "update_entity":  
            entity = input("请输入实体名称：")  
            new_properties_str = input("请输入实体的新属性（字典格式）：")  
            try:  
                new_properties = ast.literal_eval(new_properties_str)  
                if not isinstance(new_properties, dict):  
                    raise ValueError("输入必须是字典格式")  
                result = ai_nirvana.update_entity(entity, new_properties)  
                print(f"实体更新结果：{result}")  
            except (SyntaxError, ValueError) as e:  
                print(f"输入格式错误：{e}. 请确保使用正确的字典格式，例如 {{'type': 'City'}}")  
            return {"continue": True}
    
        elif command == "delete_entity":  
            entity = input("请输入要删除的实体名称：")  
            result = ai_nirvana.delete_entity(entity)  
            print(f"删除实体结果：{result}")  
            return {"continue": True}  
    
        elif command == "add_relation":  
            entity1 = input("请输入第一个实体名称：")  
            entity2 = input("请输入第二个实体名称：")  
            relation = input("请输入关系名称：")  
            try:  
                ai_nirvana.add_relationship(entity1, entity2, relation)  
                print(f"关系 {relation} 已添加，实体 {entity1} 和 {entity2} 之间。")  
            except Exception as e:  
                print(f"添加关系时发生错误：{str(e)}")  
            return {"continue": True}
    
        elif command == "get_graph_summary":  
            summary = ai_nirvana.get_graph_summary()  
            print(f"图摘要: {summary}")  
            return {"continue": True}
    
        elif command == "export_graph":  
            format = input("请输入导出格式（graphml/gexf）：")  
            result = ai_nirvana.export_graph(format)  
            print(result)  
            return {"continue": True}  # 添加这行
    
        elif command == "infer_commonsense":  
            context = input("请输入推理上下文：")  
            inference = ai_nirvana.infer_commonsense(context)  
            if inference:  
                print(f"常识推理结果: {inference}")  
            else:  
                print("无法进行常识推理。请尝试不同的上下文。")  
            return {"continue": True}
    
        elif command == "retrieve_knowledge":
            ai_nirvana.interactive_knowledge_retrieval()
            return {"continue": True}
    
        elif command == "semantic_search":
            query = input("请输入搜索内容：")
            k = int(input("请输入返回结果的数量："))
            ai_nirvana.semantic_search(query, k)

        elif command == "get_entity_info":
            entity = input("请输入实体名称：")
            result = ai_nirvana.get_entity_info(entity)
            if result:
                print(f"实体 '{entity}' 的信息: {result}")
            else:
                print(f"未找到实体 '{entity}' 的信息")
            return {"continue": True}
        
        elif command == "get_related_entities":
            entity = input("请输入实体名称：")
            ai_nirvana.get_related_entities(entity)

        elif command == "get_all_entities":
            ai_nirvana.get_all_entities()

        elif command == "get_all_relationships":
            ai_nirvana.get_all_relationships()

        elif command == "run_agent":
            query = input("请输入查询内容：")
            ai_nirvana    

        else:
            response = ai_nirvana.process(command)
            print_user_input(command)
            print("\n回答：")
            print_assistant_response(response)
            print_dialogue_context(ai_nirvana.dialogue_manager.get_dialogue_context())
            return {"continue": True}

    except Exception as e:  
        logger.error(f"Error handling command '{command}': {str(e)}")  
        return {"message": "处理命令时发生错误，请重试或联系支持。", "continue": True}  