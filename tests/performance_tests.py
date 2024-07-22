# tests/performance_tests.py  

import sys  
import os  
import time  
import cProfile  
import pstats  

# 将项目根目录添加到 Python 路径  
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

from src.core import (  
    langchain, langgraph, langsmith, lora,   
    active_learning, auto_feature_engineering,  
    digital_twin, generative_ai, intelligent_agent,  
    model_interpretability, multimodal, privacy_enhancement,  
    quantization, reinforcement_learning, semi_supervised_learning  
)  

def test_module_performance(module, function_name, *args, **kwargs):  
    start_time = time.time()  
    profiler = cProfile.Profile()  
    profiler.enable()  

    # 执行函数  
    if isinstance(module, type):  
        instance = module()  
        result = getattr(instance, function_name)(*args, **kwargs)  
    else:  
        result = getattr(module, function_name)(*args, **kwargs)  

    profiler.disable()  
    end_time = time.time()  

    # 打印执行时间  
    module_name = module.__name__ if hasattr(module, '__name__') else module.__class__.__name__  
    print(f"{module_name}.{function_name} 执行时间: {end_time - start_time:.4f} 秒")  

    # 输出性能分析结果  
    stats = pstats.Stats(profiler).sort_stats('cumulative')  
    stats.print_stats(10)  # 打印前10个最耗时的函数  

    return result  

def main():  
    # 测试 LangChain  
    print("Testing LangChain get_response (first call):")  
    test_module_performance(langchain, 'get_response', "What is AI?")  
    print("\nTesting LangChain get_response (second call, should be cached):")  
    test_module_performance(langchain, 'get_response', "What is AI?")  
    print("\n" + "="*50 + "\n")  

    # 测试 LangGraph  
    graph = langgraph.LangGraph()
    print("\nTesting LangGraph retrieve_knowledge (first call):")
    test_module_performance(graph, 'retrieve_knowledge', "Who invented the telephone?")
    print("\nTesting LangGraph retrieve_knowledge (second call, should be cached):")
    test_module_performance(graph, 'retrieve_knowledge', "Who invented the telephone?")
    
    print("\nTesting LangGraph reason:")
    test_module_performance(graph, 'reason', "All birds can fly", "Penguins can fly")
    
    print("\nTesting LangGraph infer_commonsense:")
    test_module_performance(graph, 'infer_commonsense', "It's raining outside and John doesn't have an umbrella")

    # 测试 LangSmith
    smith = langsmith.LangSmith()
    print("\nTesting LangSmith generate_code (first call):")
    test_module_performance(smith, 'generate_code', "Write a Python function to calculate fibonacci sequence")
    print("\nTesting LangSmith generate_code (second call, should be cached):")
    test_module_performance(smith, 'generate_code', "Write a Python function to calculate fibonacci sequence")

    print("\nTesting LangSmith refactor_code (first call):")
    test_module_performance(smith, 'refactor_code', "def f(x):\n    if x == 0:\n        return 1\n    else:\n        return x * f(x-1)")
    print("\nTesting LangSmith refactor_code (second call, should be cached):")
    test_module_performance(smith, 'refactor_code', "def f(x):\n    if x == 0:\n        return 1\n    else:\n        return x * f(x-1)")

    print("\nTesting LangSmith translate_text (first call):")
    test_module_performance(smith, 'translate_text', "Hello, world!", "French")
    print("\nTesting LangSmith translate_text (second call, should be cached):")
    test_module_performance(smith, 'translate_text', "Hello, world!", "French")

if __name__ == "__main__":
    main()