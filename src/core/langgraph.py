import os  
import spacy
from functools import lru_cache  
from typing import Any, Dict, List, Tuple  
from unittest.mock import Mock  

import networkx as nx  
from dotenv import load_dotenv  
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

import openai  

load_dotenv()  

class APIConfig:  
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo-0125")  
    API_KEY = os.getenv("API_KEY")  
    API_BASE = os.getenv("API_BASE", "https://api.gptsapi.net/v1")  
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))  
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))  

openai.api_base = APIConfig.API_BASE  
openai.api_key = APIConfig.API_KEY  

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
        print(f"API_KEY: {APIConfig.API_KEY[:5]}...{APIConfig.API_KEY[-5:]}")  
        print(f"API_BASE: {APIConfig.API_BASE}")  
        print(f"MODEL_NAME: {APIConfig.MODEL_NAME}")  

        # 预先定义所有属性  
        self._entity_extraction_chain = None  
        self._inference_chain = None  
        self._reasoning_chain = None  
        self._agent = None  
        self._vector_store = None  
        self.qa_chain = None  

        # 初始化核心组件  
        self.llm = ChatOpenAI(  
            temperature=APIConfig.TEMPERATURE,  
            model_name=APIConfig.MODEL_NAME,  
            openai_api_key=APIConfig.API_KEY,  
            openai_api_base=APIConfig.API_BASE  
        )  
        self.embeddings = OpenAIEmbeddings(  
            model="text-embedding-ada-002",  
            openai_api_key=APIConfig.API_KEY,  
            openai_api_base=APIConfig.API_BASE,    
        )  
        self._entity_extraction_chain = self._create_entity_extraction_chain()  
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)  
        self.nlp = spacy.load("en_core_web_sm")  
        self.graph = ExtendedNetworkxEntityGraph()  # 使用扩展的图形类
        self._initialize_components()  

        print("Testing embeddings...")  
        test_embedding = self.embeddings.embed_query("Hello, world!")  
        print(f"Embedding size: {len(test_embedding)}")  

    def _initialize_components(self):  
        # 初始化其他组件  
        self._inference_chain = self._setup_inference_chain()  
        self._vector_store = FAISS.from_texts(["Your initial texts here"], embedding=self.embeddings)  
        self.qa_chain = RetrievalQA.from_chain_type(  
            llm=self.llm,  
            chain_type="stuff",  
            retriever=self._vector_store.as_retriever(),  
            return_source_documents=True  
        )  
        self._agent = self._setup_agent()  

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

    def update_entity(self, entity, attributes):  
        if entity in self.graph.get_networkx_graph().nodes:  
            self.graph.add_node(entity, **attributes)  

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

    def retrieve_knowledge(self, query: str) -> str:
        return self._cached_run(query)

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
        self.graph.add_node(entity, type=entity_type)  
        self.vector_store.add_texts([f"{entity}: Type - {entity_type}"])  

    def add_relationship(self, entity1: str, entity2: str, relationship: str):
        self.graph.add_edge(entity1, entity2, relationship=relationship)
        self.vector_store.add_texts([f"{entity1} {relationship} {entity2}"])

    def get_entity_info(self, entity: str) -> Dict[str, Any]:
        return dict(self.graph.get_networkx_graph().nodes[entity])

    def get_related_entities(self, entity: str) -> List[str]:
        return list(self.graph.get_networkx_graph().neighbors(entity))

    def semantic_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        try:
            return self.vector_store.similarity_search_with_score(query, k=k)
        except Exception as e:
            print(f"Error in semantic_search: {str(e)}")
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
        return None 

    def delete_entity(self, entity: str):  
        if entity in self.graph.get_networkx_graph().nodes():  
            self.graph.get_networkx_graph().remove_node(entity)  
            return f"Entity '{entity}' deleted from the graph."  
        return None 

    def get_all_entities(self) -> List[str]:  
        return list(self.graph.get_networkx_graph().nodes())

    def get_all_relationships(self) -> List[Tuple[str, str, str]]:
        return [(u, v, d.get('relationship', '')) for u, v, d in self.graph.get_networkx_graph().edges(data=True)]
    
    def summary(self):  
        return self.graph.summary()
        
# 测试代码  
if __name__ == "__main__":  
    test_instance = LangGraph()  
    print("Inference chain:", test_instance.inference_chain)