import os
import io
import sys
import argparse
import logging
import time
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

logger = logging.getLogger(__name__)

# Flask server code
app = Flask(__name__)

# 在文件的适当位置定义 AVAILABLE_COMMANDS
AVAILABLE_COMMANDS = [
    'quit', 'clear', 'sentiment', 'execute', 'validate_code', 'supported_languages', 'summarize', 'change_model', 
    'vars', 'explain', 'encrypt', 'decrypt', 'add_knowledge', 'get_knowledge', 
    'list_knowledge', 'extract_keywords', 'plan_task', 'translate', 'help',
    'test_complex', 'test_models', 'test_code_safety', 'qa','init_active_learner',
    'run_active_learning', 'al_model', 'al_committee', 'al_plot','label_initial_data',
    'view_committee','init_feature_engineer', 'create_entity_set', 'generate_features',
    'get_important_features', 'remove_low_info_features','remove_correlated_features', 'create_custom_feature',
    'get_feature_types', 'get_feature_descriptions','normalize_features', 'encode_categorical_features',
    'create_digital_twin', 'simulate_digital_twin', 'monitor_digital_twin',
    'optimize_digital_twin', 'update_digital_twin_model', 'validate_digital_twin_model', 'generate_text',
    'classify_image', 'caption_image', 'fine_tune_model', 'save_model', 'load_model','create_agent', 'train_agent',
    'run_agent', 'setup_rl_agent', 'rl_decide',
]
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
        except Exception as e:
            print(f"数字孪生系统初始化失败: {e}")

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
        if model_type == 'general':
            # 保存通用模型（如果有的话）
            self.generative_ai.save_model(filename)
        elif model_type == 'agent':
            # 保存智能代理模型
            if hasattr(self, 'agent'):
                self.agent.save(filename)
            else:
                raise ValueError("No agent model to save.")
        else:
            raise ValueError("Invalid model type. Use 'general' or 'agent'.")
        print(f"Saved {model_type} model to {filename}")

    def load_model(self, filename: str, model_type: str = 'general'):
        if model_type == 'general':
            # 加载通用模型
            self.generative_ai.load_model(filename)
        elif model_type == 'agent':
            # 加载智能代理模型
            if hasattr(self, 'agent'):
                self.agent.load(filename)
            else:
                raise ValueError("No agent initialized. Use 'create_agent' first.")
        else:
            raise ValueError("Invalid model type. Use 'general' or 'agent'.")
        print(f"Loaded {model_type} model from {filename}")
        
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

    def rl_decide(self, state):
        if not hasattr(self, 'rl_agent'):
            raise ValueError("RL agent not set up. Use 'setup_rl_agent' first.")
        action = self.rl_agent.act(state)
        return action         

    def check_agent_attributes(self):
        if hasattr(self, 'agent'):
            print("DQNAgent attributes:")
            for attr in dir(self.agent):
                if not attr.startswith('__'):
                    print(f"- {attr}")
        else:
            print("Agent has not been created yet.") 
    
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
    try:
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
        elif command == "help":
            print_help()
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
            ai_nirvana.knowledge_base.add_knowledge(key, value)
            return {"message": f"知识已添加: {key}", "continue": True}
        elif command == "get_knowledge":
            key = input("请输入要检索的知识键：")
            try:
                value = ai_nirvana.knowledge_base.get_knowledge(key)
                print(f"知识内容：{value}")
            except KeyError as e:
                print(f"错误：{str(e)}")
            return {"continue": True}
        elif command == "list_knowledge":
            all_knowledge = ai_nirvana.knowledge_base.list_all_knowledge()
            print("所有知识：")
            for k, v in all_knowledge.items():
                print(f"{k}: {v}")
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
            try:
                state = input("Enter the current state (comma-separated values): ")
                state = [float(x.strip()) for x in state.split(',')]
                action = ai_nirvana.rl_decide(state)
                return {"message": f"RL agent decided on action: {action}", "continue": True}
            except Exception as e:
                return {"message": f"Error in RL decision: {str(e)}", "continue": True}

        else:
            response = ai_nirvana.process(command)
            print_user_input(command)
            print("\n回答：")
            print_assistant_response(response)
            print_dialogue_context(ai_nirvana.dialogue_manager.get_dialogue_context())
            return {"continue": True}

    except Exception as e:
        logger.error(f"处理命令时发生未预期的错误: {str(e)}")
        print(f"发生错误: {str(e)}。如果问题持续，请联系系统管理员。")
        return {"continue": True}
    
def print_help() -> None:
    print("\n可用命令：")
    for cmd in AVAILABLE_COMMANDS:
        description = {
            'quit': '退出程序',
            'clear': '清除对话历史和变量状态',
            'sentiment': '对下一条输入进行情感分析',
            'execute': '执行 Python 代码',
            'validate_code': '验证 Python 代码的安全性',
            'supported_languages': '显示支持的编程语言列表',
            'summarize': '生成文本摘要',
            'change_model': '更改使用的模型',
            'vars': '显示当前定义的所有变量',
            'explain': '解释模型预测',
            'encrypt': '加密敏感数据',
            'decrypt': '解密数据',
            'add_knowledge': '添加知识到知识库',
            'get_knowledge': '检索知识库中的知识',
            'list_knowledge': '列出所有知识库中的知识',
            'extract_keywords': '从文本中提取关键词',
            'plan_task': '为给定任务生成计划',
            'translate': '翻译文本到指定语言',
            'help': '显示此帮助信息',
            'test_complex': '执行复杂文本分析测试',
            'test_models': '执行模型比较测试',
            'test_code_safety': '执行代码安全性测试',
            'qa': '问答功能',
            'init_active_learner': '初始化主动学习器',
            'run_active_learning': '执行主动学习循环',
            'al_model': '获取或设置主动学习模型',
            'al_committee': '创建主动学习委员会',
            'al_plot': '绘制主动学习曲线',
            'label_initial_data': '标记数据',
            'view_committee': '再次查看委员会',
            'init_feature_engineer': '初始化特征工程器',
            'create_entity_set': '创建实体集',
            'generate_features': '生成特征',
            'get_important_features': '获取重要特征',
            'remove_low_info_features': '移除低信息特征',
            'remove_correlated_features': '移除高度相关特征',
            'create_custom_feature': '创建自定义特征',
            'get_feature_types': '获取特征类型',
            'get_feature_descriptions': '获取特征描述',
            'normalize_features': '标准化特征',
            'encode_categorical_features': '编码分类特征',
            'create_digital_twin': '创建一个新的数字孪生系统',
            'simulate_digital_twin': '模拟数字孪生系统状态变化',
            'monitor_digital_twin': '监控数字孪生系统状态并检测异常',
            'optimize_digital_twin': '优化数字孪生系统参数',
            'update_digital_twin_model': '更新数字孪生系统的物理模型',
            'validate_digital_twin_model': '验证数字孪生系统模型的准确性',
            'generate_text': '生成文本',
            'classify_image': '对图像进行分类',
            'caption_image': '为图像生成描述',
            'fine_tune_model': '微调模型',
            'save_model': '保存模型（通用模型或智能代理模型)',
            'load_model': '加载模型（通用模型或智能代理模型)',
            'create_agent' : '加 创建一个新的 DQN 智能代理',
            'train_agent' : '在指定环境中训练智能代理',
            'run_agent' : '在指定环境中运行智能代理',
            'setup_rl_agent' : '设置强化学习代理',
            'rl_decide' : '让强化学习代理根据给定状态做出决策',

        }    
        print(f"'{cmd}' - {description.get(cmd, '暂无描述')}")

    print("\n注意：")
    print("- 执行代码时，某些操作（如文件操作和模块导入）出于安全考虑是受限的。")
    print("- 支持基本的Python操作，包括变量赋值、条件语句、循环等。")
    print("- 如果遇到'未定义'的错误，可能是因为该操作被安全限制所阻止。")
    print("- 请按正确顺序使用主动学习相关命令：先初始化主动学习器，再执行主动学习。")
    print("- 主动学习可能涉及大量数据和计算，请确保有足够的系统资源。")
    print("- 长时间运行的操作可以通过 Ctrl+C 中断。")
    print("'init_feature_engineer' - 初始化特征工程器")
    print("  - 支持使用示例数据或本地文件")
    print("  - 使用示例数据时可以指定样本数量和特征数量")
    print("  - 使用本地文件时需要提供文件路径和目标列名称")
    print("'get_important_features' - 获取最重要的特征（需要先初始化特征工程器）")
    print("- 在使用 `create_digital_twin` 命令时，请确保物理系统模型函数定义正确且符合要求。")
    print("- 在进行模拟、监控和优化操作前，请确认数字孪生系统已经成功创建。")
    print("- 使用 `update_digital_twin_model` 更新物理模型时，请谨慎操作，确保新模型的准确性和稳定性。")
    print("- 进行大规模模拟和优化任务时，请注意系统资源的使用，避免因资源不足导致的性能问题。")
    print("- 在处理复杂的物理系统模型时，建议进行充分的测试，以确保系统行为符合预期。")
    print("- 数字孪生系统的各项功能依赖于输入数据的准确性，请确保输入的初始条件、时间步长、传感器数据等信息的准确性。")
    print("- 'generate_text' 命令用于生成文本，可以指定最大令牌数和温度参数。")
    print("- 'classify_image' 和 'caption_image' 命令需要提供有效的图像文件路径。")
    print("- 'fine_tune_model' 命令用于微调模型，需要提供训练数据和相关参数。")
    print("- 使用 'save_model' 和 'load_model' 命令时，请确保指定正确的文件路径。")
    print("- 图像处理和模型微调功能可能需要较长时间，请耐心等待。")
    print("- 在使用模型相关功能时，请确保系统有足够的计算资源。")
    print("- 'save_model' 和 'load_model' 命令现在可以处理both通用模型和智能代理模型。使用时请指定模型类型（'general' 或 'agent'）。")
    print("- 在使用智能代理相关功能（如 'create_agent'、'train_agent'）之前，请确保已安装必要的依赖，如 TensorFlow 和 OpenAI Gym。")
    print("- 智能代理的训练可能需要较长时间，请耐心等待。训练过程中会显示进度信息。")
    print("- 在使用 'rl_decide' 命令时，请确保先使用 'setup_rl_agent' 设置了强化学习代理。")
    print("- 如果遇到任何未预期的错误或异常行为，请检查日志文件以获取更详细的信息。")
    print("- 定期保存您的工作成果和模型，以防意外情况发生。")
    print("- 如果您是在共享环境中使用本系统，请注意保护敏感数据和模型的安全。")
    
def main(config: Config):  
    ai_nirvana = AINirvana(config)  
    print("欢迎使用 AI Nirvana 智能助手！")  
    print("输入 'help' 查看可用命令。")  

    while True:  
        try:  
            print("\n请输入您的问题或命令（输入空行发送）：")  
            user_input = input().strip()  
            if not user_input:  
                print("您似乎没有输入任何内容。请输入一些文字或命令。")  
                continue  

            print("正在处理命令...")  
            result = handle_command(user_input, ai_nirvana)  
            print(f"命令处理完成，结果: {result}")  # 修改  

            if not result.get("continue", True):  
                print(result.get("message", "再见！"))  
                break  

            print("准备接收下一个命令")  

        except Exception as e:  
            print(f"发生错误: {str(e)}")  
            print("错误详情:")  
            import traceback  
            traceback.print_exc()  # 打印详细的错误堆栈  

    print("感谢使用 AI Nirvana 智能助手，再见！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Nirvana Assistant")
    parser.add_argument("--config", default="config.json", help="Path to the configuration file")
    args = parser.parse_args()

    try:
        config = Config(args.config)
        if not config.validate_config():
            logger.error("Configuration validation failed. Please check your config file.")
            sys.exit(1)

        main(config)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

# Flask server code
app = Flask(__name__)

@app.route("/process", methods=["POST"])
def process():
    input_text = request.json.get("input")
    config = Config("config.json")
    ai_nirvana = AINirvana(config)
    response = ai_nirvana.process(input_text)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)