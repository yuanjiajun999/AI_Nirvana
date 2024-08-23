import sys  
import os  

# 获取当前文件的绝对路径  
current_file_path = os.path.abspath(__file__)  

# 获取项目根目录路径（假设 test_langgraph_integration.py 在项目根目录）  
project_root = os.path.dirname(current_file_path)  

# 将 src 目录添加到 Python 路径  
src_path = os.path.join(project_root, 'src')  
sys.path.append(src_path)  

# 现在尝试导入 LangGraph  
from core.langgraph import LangGraph  
from dotenv import load_dotenv  

# 打印 Python 路径以进行调试  
print("Python path:")  
for path in sys.path:  
    print(path)  

# 加载 .env 文件中的环境变量  
load_dotenv()  

# 从环境变量中获取配置  
api_key = os.getenv("API_KEY")  
api_base = os.getenv("API_BASE")  
model_name = os.getenv("MODEL_NAME")  

# 确保必要的环境变量已设置  
if not api_key or not api_base or not model_name:  
    raise ValueError("请确保 API_KEY, API_BASE 和 MODEL_NAME 已在 .env 文件中正确设置")  

# 初始化 LangGraph  
# 注意：LangGraph 的 __init__ 方法不接受参数，所以我们移除了之前的参数  
lang_graph = LangGraph()  

lang_graph = LangGraph()  
print("Testing agent...")  
response = lang_graph.run_agent("Tell me about Apple")  
print(f"Agent response: {response}")

print("LangGraph initialized successfully.")  

# 测试一个简单的方法调用  
try:  
    entities = lang_graph.extract_entities("Apple is a technology company based in Cupertino.")  
    print(f"Extracted entities: {entities}")  
except Exception as e:  
    print(f"Error occurred while extracting entities: {str(e)}")  

# 测试知识图谱操作  
try:  
    lang_graph.add_entity("Apple", {"type": "Company", "industry": "Technology"})  
    print("Entity 'Apple' added successfully.")  
    
    related_entities = lang_graph.get_related_entities("Apple")  
    print(f"Related entities to 'Apple': {related_entities}")  
except Exception as e:  
    print(f"Error occurred during knowledge graph operations: {str(e)}")  

# 测试语义搜索  
try:  
    search_results = lang_graph.semantic_search("technology companies", k=3)  
    print(f"Semantic search results: {search_results}")  
except Exception as e:  
    print(f"Error occurred during semantic search: {str(e)}")  

# 测试 agent 运行  
try:  
    agent_response = lang_graph.run_agent("Tell me about Apple company")  
    print(f"Agent response: {agent_response}")  
except Exception as e:  
    print(f"Error occurred while running agent: {str(e)}")  

# 获取图谱摘要  
try:  
    graph_summary = lang_graph.get_graph_summary()  
    print(f"Graph summary: {graph_summary}")  
except Exception as e:  
    print(f"Error occurred while getting graph summary: {str(e)}")  

# 测试实体更新  
try:  
    update_result = lang_graph.update_entity("Apple", {"headquarters": "Cupertino"})  
    print(f"Entity update result: {update_result}")  
except Exception as e:  
    print(f"Error occurred while updating entity: {str(e)}")  

# 获取所有实体  
try:  
    all_entities = lang_graph.get_all_entities()  
    print(f"All entities in the graph: {all_entities}")  
except Exception as e:  
    print(f"Error occurred while getting all entities: {str(e)}")    