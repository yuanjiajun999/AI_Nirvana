import os  
import openai
import logging
import spacy
from functools import lru_cache  
from typing import Any, Dict, List, Tuple  
from unittest.mock import Mock  
import json
import networkx as nx   
from openai import OpenAIError  
from langchain.chains import RetrievalQA, LLMChain  
from langchain.agents import AgentType, AgentExecutor, create_react_agent  
from langchain.chains import GraphQAChain  
from langchain.memory import ConversationBufferMemory  
from langchain.prompts import PromptTemplate  
from langchain.schema import AIMessage, HumanMessage  
from langchain_community.graphs import NetworkxEntityGraph  
from langchain_community.vectorstores import FAISS  
from langchain_core.tools import Tool  
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) 

class APIConfig:  
    def __init__(self, config: Dict[str, Any]):
        self.MODEL_NAME = config.get("model", "gpt-3.5-turbo-0125")
        self.API_KEY = config.get("api_key")
        self.API_BASE = config.get("api_base", "https://api.gptsapi.net/v1")
        self.TEMPERATURE = float(config.get("temperature", "0.7"))
        self.MAX_TOKENS = int(config.get("max_tokens", "256"))

def load_config(file_path: str = "config.json") -> Dict[str, Any]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

config = load_config()  # 加载 config.json 中的配置

api_config = APIConfig(config)

openai.api_base = api_config.API_BASE  
openai.api_key = api_config.API_KEY  
class ExtendedNetworkxEntityGraph(NetworkxEntityGraph):  
    def __init__(self):  
        super().__init__()  
        self._graph = nx.Graph()  

    def add_node(self, node_id, **attr):  
        self._graph.add_node(node_id, **attr)  

    def add_edge(self, node1, node2, **attr):  
        self._graph.add_edge(node1, node2, **attr)  

    def get_networkx_graph(self):  
        return self._graph

    def summary(self):  
        return {  
            "num_nodes": self._graph.number_of_nodes(),  
            "num_edges": self._graph.number_of_edges(),
            "nodes": list(self._graph.nodes(data=True)),
            "edges": list(self._graph.edges(data=True)),
        }   

class LangGraph:  
    def __init__(self):  
        print(f"API_KEY: {api_config.API_KEY[:5]}...{api_config.API_KEY[-5:]}")  
        print(f"API_BASE: {api_config.API_BASE}")  
        print(f"MODEL_NAME: {api_config.MODEL_NAME}")  

        # 预先定义所有属性  
        self._entity_extraction_chain = None  
        self._inference_chain = None  
        self._reasoning_chain = None  
        self._agent = None  
        self._vector_store = None  
        self.qa_chain = None  
        self.use_fallback = False
        self.fallback_knowledge = {}
        self._vector_store = None  # 初始化为 None 
        self.initialize_vector_store() 
        
        # 添加 similarity_threshold 属性，并设置一个默认值
        self.similarity_threshold = 0.8  # 这个值可以根据需要进行调整
        self.graph = ExtendedNetworkxEntityGraph()  # 使用扩展的图形类
        logger.info(f"LangGraph initialized. Graph attribute exists: {hasattr(self, 'graph')}")
        
        try:
             # 初始化核心组件  
            self.llm = ChatOpenAI(  
                temperature=api_config.TEMPERATURE,  
                model_name=api_config.MODEL_NAME,  
                openai_api_key=api_config.API_KEY,  
                openai_api_base=api_config.API_BASE  
            )  
            self.embeddings = OpenAIEmbeddings(  
                model="text-embedding-ada-002",  
                openai_api_key=api_config.API_KEY,  
                openai_api_base=api_config.API_BASE,    
            )    
            
            # 使用一些实际的初始文本
            initial_texts = ["北京是中国的首都", "北京有许多著名的历史古迹"]
            self._vector_store = FAISS.from_texts(initial_texts, embedding=self.embeddings)
            
            self._entity_extraction_chain = self._create_entity_extraction_chain()  
            self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)  
            self.nlp = spacy.load("en_core_web_sm")  
            self._initialize_components()  

            print("Testing embeddings...")  
            test_embedding = self.embeddings.embed_query("Hello, world!")  
            print(f"Embedding size: {len(test_embedding)}")  

            logging.info("LangGraph initialized successfully with FAISS.")
        except Exception as e:
            logging.error(f"Failed to initialize LangGraph: {e}")
            self.use_fallback = True
            logging.info("Using fallback mode.")
            # 在fallback模式下，我们仍然需要初始化一些基本组件
            self.fallback_knowledge = dict(zip(initial_texts, initial_texts))
            
    def _initialize_components(self):
        try:
            # 初始化其他组件  
            self._inference_chain = self._setup_inference_chain()  
            self.qa_chain = RetrievalQA.from_chain_type(  
                llm=self.llm,  
                chain_type="stuff",  
                retriever=self._vector_store.as_retriever(),  
                return_source_documents=True  
            )  
            self._agent = self._setup_agent()
        except Exception as e:
            logging.error(f"Error in _initialize_components: {e}")
            self.use_fallback = True

    def initialize_vector_store(self):  
        try:  
            # 使用适当的向量存储实现，例如 FAISS 或 Chroma  
            from langchain.vectorstores import FAISS  
            from langchain.embeddings import OpenAIEmbeddings  

            # 初始化嵌入模型  
            embeddings = OpenAIEmbeddings()  

            # 创建一个空的向量存储  
            self._vector_store = FAISS.from_texts(["初始化文档"], embeddings)  
            
            logging.info("Vector store initialized successfully")  
        except Exception as e:  
            logging.error(f"Error initializing vector store: {str(e)}", exc_info=True)  
            self._vector_store = None  
            
    def _create_entity_extraction_chain(self):  
        prompt_template = PromptTemplate(  
            input_variables=["text"],  
            template="Extract the entities from the following text:\n\n{text}\n\nEntities:"  
        )  
        return LLMChain(llm=self.llm, prompt=prompt_template) 

    def add_relation(self, entity1, entity2, relation):  
        self.graph.add_edge(entity1, entity2, relation=relation)  

    def get_graph_summary(self):  
        return self.graph.summary()  

    def get_all_entities(self):  
        return list(self.graph.get_networkx_graph().nodes(data=True))  

    def _setup_inference_chain(self):  
        return LLMChain(llm=self.llm, prompt=PromptTemplate(  
            input_variables=["context"],  
            template="Given the context: {context}, please make a common sense inference."  
        ))  

    def _setup_entity_extraction_chain(self):  
        # 实现实体提取链的逻辑  
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Extract entities from the following text:\n\n{text}\n\nEntities:"
        )
        return LLMChain(llm=self.llm, prompt=prompt)

    def _setup_knowledge_graph_qa(self):  
       # 实现知识图谱问答的逻辑  
       pass  

    def _setup_agent(self):  
        tools = [  
            Tool(name="Knowledge Graph QA", func=self.retrieve_knowledge, description="Useful for answering questions based on the knowledge graph."),  
            Tool(name="Entity Extraction", func=self.extract_entities, description="Useful for extracting entities from text."),  
            Tool(name="Reasoning", func=self.reason, description="Useful for logical reasoning tasks."),  
            Tool(name="Common Sense Inference", func=self.infer_commonsense, description="Useful for making common sense inferences.")  
        ]  

        prompt = PromptTemplate.from_template(  
            "You are an AI assistant with access to a knowledge graph and various reasoning tools. "  
            "Your task is to answer user queries by leveraging the knowledge graph and applying logical reasoning.\n"  
            "Human: {input}\n"  
            "AI: Let's approach this step-by-step:\n"  
            "1) First, I'll check if we need to extract any entities from the query.\n"  
            "2) Then, I'll search the knowledge graph for relevant information.\n"  
            "3) If needed, I'll apply reasoning or common sense inference.\n"  
            "4) Finally, I'll formulate a comprehensive answer.\n\n"  
            "Let's begin:\n\n"  
            "Available tools:\n{tools}\n\n"  
            "Use the following format:\n"  
            "Thought: Consider what to do next\n"  
            "Action: Choose an action: {tool_names}\n"  
            "Action Input: Provide the input for the action\n"  
            "Observation: The result of the action\n\n"  
            "... (this Thought/Action/Action Input/Observation can repeat N times)\n"  
            "Thought: I now know the final answer\n"  
            "Final Answer: The final answer to the original input question\n\n"  
            "{agent_scratchpad}"  
        )  

        agent = create_react_agent(self.llm, tools, prompt)  
        self._agent = AgentExecutor.from_agent_and_tools(  
            agent=agent,   
            tools=tools,   
            verbose=True,  
            handle_parsing_errors=True  # 添加此参数  
        )  
        return self._agent

    @property
    def entity_extraction_chain(self):
        return self._entity_extraction_chain

    @property
    def inference_chain(self):
        return self._inference_chain

    @property
    def reasoning_chain(self):
        return self._reasoning_chain

    @property
    def agent(self):
        return self._agent

    @property
    def vector_store(self):
        return self._vector_store
        
    @lru_cache(maxsize=100)
    def _cached_run(self, query: str) -> str:
        try:
            return self.qa_chain.invoke(query)
        except OpenAIError as e:
            print(f"API error in _cached_run: {str(e)}")
            return "Sorry, an error occurred while processing your request."

    def retrieve_knowledge(self, query: str) -> Dict[str, Any]:
        try:
            # 首先尝试从向量存储中检索
            results = self._vector_store.similarity_search_with_score(query, k=1)
            if results and results[0][1] < self.similarity_threshold:
                content, score = results[0]
                return {
                    "query": query,
                    "result": content.page_content,
                    "source": "Knowledge Base",
                    "score": score
                }
            
            # 如果没有找到相关信息，使用 AI 生成
            ai_response = self.generate_ai_response(query)
            if ai_response:
                self.add_knowledge(query, ai_response)
                return {
                    "query": query,
                    "result": ai_response,
                    "source": "AI generated"
                }
            
            return {
                "query": query,
                "result": "抱歉，无法找到或生成相关信息。",
                "source": "Not found"
            }
        except Exception as e:
            logging.error(f"Error in retrieve_knowledge: {e}")
            return {
                "query": query,
                "result": f"检索知识时发生错误: {str(e)}",
                "source": "Error"
            }
        
    def reason(self, context: str) -> str:  
        try:  
            response = self.llm.invoke(f"Given the context: {context}, please provide a logical reasoning.")  
            return response  
        except Exception as e:  
            print(f"Error in reason method: {str(e)}")  
            return "An error occurred during reasoning."

    def infer_commonsense(self, context: str) -> str:
        try:
            return self.inference_chain.run(context=context)
        except OpenAIError as e:
            print(f"API error in infer_commonsense: {str(e)}")
            return "An error occurred during inference."

    def extract_entities(self, input_data):  
        if isinstance(input_data, (AIMessage, HumanMessage)):  
            text = input_data.content  
        elif isinstance(input_data, str):  
            text = input_data  
        else:  
            raise ValueError(f"Unsupported input type: {type(input_data)}")  
        
        doc = self.nlp(text)  
        entities = [(ent.text, ent.label_) for ent in doc.ents]  
        return entities  

    def add_entity(self, entity: str, entity_type: str):
        try:
            logger.info(f"Attempting to add entity: {entity} of type: {entity_type}")
            if not hasattr(self, 'graph'):
                logger.error("Graph attribute does not exist")
                return "Error: Graph attribute does not exist"
        
            if entity not in self.graph.get_networkx_graph().nodes():
                self.graph.get_networkx_graph().add_node(entity, type=entity_type)
                logger.info(f"Entity '{entity}' of type '{entity_type}' added successfully.")
                return f"Entity '{entity}' of type '{entity_type}' added successfully."
            else:
                logger.info(f"Entity '{entity}' already exists.")
                return f"Entity '{entity}' already exists."
        except AttributeError as e:
            logger.error(f"AttributeError in add_entity: {str(e)}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in add_entity: {str(e)}")
            return f"Error: An unexpected error occurred while adding the entity."

    def add_relationship(self, entity1: str, entity2: str, relationship: str):
        self.graph.add_edge(entity1, entity2, relationship=relationship)
        self.vector_store.add_texts([f"{entity1} {relationship} {entity2}"])

    def get_entity_info(self, entity: str) -> Dict[str, Any]:
        return dict(self.graph.get_networkx_graph().nodes[entity])

    def get_related_entities(self, entity: str) -> List[str]:
        return list(self.graph.get_networkx_graph().neighbors(entity))

    def semantic_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:  
        logging.info(f"Attempting semantic search with query: {query}, k: {k}")  
        if self._vector_store is None:  
            logging.error("Vector store is None. It may not have been initialized properly.")  
            return []  
        
        try:  
            results = self._vector_store.similarity_search_with_score(query, k=k)  
            logging.info(f"Search completed. Found {len(results)} results.")  
            logging.debug(f"Raw results: {results}")  
            
            # 添加安全检查  
            safe_results = []  
            for item in results:  
                if isinstance(item, tuple) and len(item) == 2:  
                    doc, score = item  
                    if hasattr(doc, 'page_content'):  
                        safe_results.append((doc.page_content, score))  
                    else:  
                        logging.warning(f"Unexpected document format: {doc}")  
                else:  
                    logging.warning(f"Unexpected result format: {item}")  
            
            return safe_results  
        except Exception as e:  
            logging.error(f"Error in semantic_search: {str(e)}", exc_info=True)  
            return []    

    def run_agent(self, query):  
        if self._agent is None:  
            return "Agent not initialized properly."  
        try:  
            response = self._agent.invoke({"input": query})  
            return response['output']  
        except Exception as e:  
            print(f"Error in run_agent: {str(e)}")  
            return f"An error occurred: {str(e)}"

    def export_graph(self, format: str = "graphml") -> str:
        g = self.graph.get_networkx_graph()
        if format == "graphml":
            nx.write_graphml(g, "graph.graphml")
            return "Graph exported as graph.graphml"
        elif format == "gexf":
            nx.write_gexf(g, "graph.gexf")
            return "Graph exported as graph.gexf"
        else:
            return "Unsupported format"

    def update_entity(self, entity: str, new_properties: dict):  
        if entity in self.graph.get_networkx_graph().nodes():  
            self.graph.get_networkx_graph().nodes[entity].update(new_properties)  
            return f"Entity '{entity}' updated successfully."  
        return f"Entity '{entity}' not found."  

    def delete_entity(self, entity: str):  
        if entity in self.graph.get_networkx_graph().nodes():  
            self.graph.get_networkx_graph().remove_node(entity)  
            return f"Entity '{entity}' deleted from the graph."  
        return f"Entity '{entity}' not found."  

    def get_all_entities(self) -> List[str]:  
        return list(self.graph.get_networkx_graph().nodes())

    def get_all_relationships(self) -> List[Tuple[str, str, str]]:
        return [(u, v, d.get('relationship', '')) for u, v, d in self.graph.get_networkx_graph().edges(data=True)]
    
    def summary(self):  
        return self.graph.summary()

    def add_knowledge(self, key: str, value: str) -> Dict[str, Any]:
        try:
            if not self.use_fallback:
                self._vector_store.add_texts([f"{key}: {value}"])
            self.fallback_knowledge[key] = value
            logging.info(f"Knowledge added: {key}")
            return {"success": True, "message": f"Knowledge '{key}' added successfully"}
        except Exception as e:
            logging.error(f"Error adding knowledge: {e}")
            return {"success": False, "message": f"Error adding knowledge: {str(e)}"}

    def query_knowledge(self, query: str) -> Dict[str, Any]:
        try:
            results = self._vector_store.similarity_search_with_score(query, k=1)
            if results:
                content, score = results[0]
                return {"found": True, "content": content.page_content, "score": score}
            return {"found": False, "message": "No relevant knowledge found"}
        except Exception as e:
            logger.error(f"Error querying knowledge: {str(e)}")
            return {"error": f"Error querying knowledge: {str(e)}"}

    def update_knowledge(self, key: str, value: str) -> Dict[str, Any]:
        try:
            # 首先删除旧的知识（如果存在）
            self.delete_knowledge(key)
            # 然后添加新的知识
            return self.add_knowledge(key, value)
        except Exception as e:
            logger.error(f"Error updating knowledge: {str(e)}")
            return {"success": False, "message": f"Error updating knowledge: {str(e)}"}

    def delete_knowledge(self, key: str) -> Dict[str, Any]:
        try:
            # 从图中删除节点
            self.graph.remove_node(key)
            # 注意：从 FAISS 向量存储中删除特定项是不直接支持的
            # 我们可以考虑重建向量存储，但这可能会很耗时
            logger.info(f"Knowledge deleted: {key}")
            return {"success": True, "message": f"Knowledge '{key}' deleted successfully"}
        except Exception as e:
            logger.error(f"Error deleting knowledge: {str(e)}")
            return {"success": False, "message": f"Error deleting knowledge: {str(e)}"}

    def list_all_knowledge(self) -> List[str]:
        return list(self.graph.nodes())

    def generate_ai_response(self, query: str) -> str:
        try:
            response = self.llm.invoke(query)
            return response.content
        except Exception as e:
            logging.error(f"Error generating AI response: {e}")
            return ""
        
# 测试代码  
if __name__ == "__main__":  
    test_instance = LangGraph()  
    print("Inference chain:", test_instance.inference_chain)