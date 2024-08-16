import os
import io
import sys
import argparse
import logging
import time
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

logger = logging.getLogger(__name__)
# 在文件的适当位置定义 AVAILABLE_COMMANDS
AVAILABLE_COMMANDS = [
    'quit', 'clear', 'sentiment', 'execute', 'summarize', 'change_model', 
    'vars', 'explain', 'encrypt', 'decrypt', 'add_knowledge', 'get_knowledge', 
    'list_knowledge', 'extract_keywords', 'plan_task', 'translate', 'help',
    'test_complex', 'test_models', 'test_code_safety', 'qa','init_active_learner',
    'run_active_learning', 'al_model', 'al_committee', 'al_plot','label_initial_data',
    'view_committee','init_feature_engineer', 'create_entity_set', 'generate_features',
    'get_important_features', 'remove_low_info_features','remove_correlated_features', 'create_custom_feature',
    'get_feature_types', 'get_feature_descriptions','normalize_features', 'encode_categorical_features'
]
# 加载 .env 文件
load_dotenv()

class AINirvana:
    def __init__(self, config: Config):
        # 注册模型
        ModelFactory.register_model("LanguageModel", LanguageModel)
        
        self.code_executor = CodeExecutor()
        self.config = config
        self.model_name = self.config.get('model', 'gpt-3.5-turbo')
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
            return "执行被阻止：不安全的代码"  

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
                return f"执行结果:\n{output}"  
            elif error:  
                return f"执行过程中出现错误:\n{error}"  
            else:  
                # 如果没有输出，显示所有定义的变量  
                var_output = []  
                for var, value in local_vars.items():  
                    if not var.startswith('__'):  
                        var_output.append(f"{var} = {value}")  
                if var_output:  
                    return "定义的变量:\n" + "\n".join(var_output)  
                else:  
                    # 如果没有定义新变量，尝试获取最后一个表达式的值  
                    last_line = code.strip().split('\n')[-1]  
                    try:  
                        result = eval(last_line, globals(), local_vars)  
                        return f"最后一个表达式的值: {result}"  
                    except:  
                        return "代码执行成功，但没有输出或定义新变量。"  
        except Exception as e:  
            return f"代码执行时发生错误: {str(e)}"  
        finally:  
            # 恢复标准输出和标准错误  
            sys.stdout = sys.__stdout__  
            sys.stderr = sys.__stderr__   
        
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
            result = ai_nirvana.execute_code(code)  
    
            # 在这里添加新的输出格式  
            print("\n" + "="*40)  
            print("执行结果:")  
            print(result)  
            print("="*40 + "\n")  
    
            explanation = input("需要进一步解释结果吗？(yes/no): ")  
            if explanation.lower() == 'yes':  
                explanation_result = ai_nirvana.explain_code_result(code, result)  
                print("解释：", explanation_result)  
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