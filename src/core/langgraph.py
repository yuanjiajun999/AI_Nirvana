import os
from functools import lru_cache
from typing import Any, Dict, List, Tuple
import networkx as nx

from dotenv import load_dotenv
from langchain.chains import GraphQAChain
from unittest.mock import Mock
from langchain.prompts import PromptTemplate
from langchain_community.graphs import NetworkxEntityGraph
from langchain_openai import ChatOpenAI
from openai import OpenAIError
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import Tool

load_dotenv()

class APIConfig:
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo-0125")
    API_KEY = os.getenv("API_KEY")
    API_BASE = os.getenv("API_BASE", "https://api.gptsapi.net/v1")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))

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

class LangGraph:
    def __init__(self):
        self._entity_extraction_chain = None
        self._inference_chain = None
        self._reasoning_chain = None
        self._agent = None
        self._vector_store = None
        self._initialize_components()

    def _initialize_components(self):
        try:
            self.llm = ChatOpenAI(
                model_name=APIConfig.MODEL_NAME,
                openai_api_key=APIConfig.API_KEY,
                openai_api_base=APIConfig.API_BASE,
                temperature=APIConfig.TEMPERATURE,
                max_tokens=APIConfig.MAX_TOKENS,
            )
            self.graph = ExtendedNetworkxEntityGraph()
            self.embeddings = OpenAIEmbeddings()
            self._vector_store = FAISS.from_texts(["Initial text"], embedding=self.embeddings)
            self.memory = ConversationBufferMemory(memory_key="chat_history")

            self._setup_chains()
            self._setup_agent()
        except OpenAIError as e:
            print(f"API error during initialization: {str(e)}")
            raise

    def _setup_chains(self):
        entity_extraction_prompt = PromptTemplate(
            template="Extract entities from the following text:\n\n{text}\n\nEntities:",
            input_variables=["text"],
        )
        self._entity_extraction_chain = entity_extraction_prompt | self.llm

        self.qa_chain = GraphQAChain.from_llm(
            llm=self.llm, graph=self.graph, verbose=True
        )

        self._reasoning_chain = self._create_reasoning_chain()
        self._inference_chain = self._create_inference_chain()

    def _create_reasoning_chain(self):
        return Mock()  # 暂时用 Mock 对象替代

    def _create_inference_chain(self):
        return Mock()  # 暂时用 Mock 对象替代

    def _setup_agent(self):
        tools = [
            Tool(
                name="Knowledge Graph QA",
                func=self.retrieve_knowledge,
                description="Useful for answering questions based on the knowledge graph."
            ),
            Tool(
                name="Entity Extraction",
                func=self.extract_entities,
                description="Useful for extracting entities from text."
            ),
            Tool(
                name="Reasoning",
                func=self.reason,
                description="Useful for logical reasoning tasks."
            ),
            Tool(
                name="Common Sense Inference",
                func=self.infer_commonsense,
                description="Useful for making common sense inferences."
            )
        ]

        self._agent = AgentExecutor.from_agent_and_tools(
            agent=create_react_agent(self.llm, tools),
            tools=tools,
            verbose=True,
            memory=self.memory
        )

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

    def reason(self, premise: str, conclusion: str) -> str:
        try:
            return self.reasoning_chain.run(premise=premise, conclusion=conclusion)
        except OpenAIError as e:
            print(f"API error in reason: {str(e)}")
            return "An error occurred during reasoning."

    def infer_commonsense(self, context: str) -> str:
        try:
            return self.inference_chain.run(context=context)
        except OpenAIError as e:
            print(f"API error in infer_commonsense: {str(e)}")
            return "An error occurred during inference."

    def extract_entities(self, text: str) -> List[str]:
        try:
            response = self.entity_extraction_chain.invoke({"text": text})
            return [entity.strip() for entity in response.split(',')]
        except OpenAIError as e:
            print(f"API error in extract_entities: {str(e)}")
            return []

    def add_entity(self, entity: str, properties: Dict[str, Any]):
        self.graph.add_node(entity, **properties)
        self.vector_store.add_texts([f"{entity}: {str(properties)}"])

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

    def run_agent(self, query: str) -> str:
        try:
            return self.agent.run(query)
        except Exception as e:
            print(f"Error in run_agent: {str(e)}")
            return "An error occurred while processing your request."

    def get_graph_summary(self) -> Dict[str, Any]:
        g = self.graph.get_networkx_graph()
        return {
            "num_nodes": g.number_of_nodes(),
            "num_edges": g.number_of_edges(),
            "density": nx.density(g),
            "connected_components": nx.number_connected_components(g),
        }

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

    def update_entity(self, entity: str, new_properties: Dict[str, Any]):
        if entity in self.graph.get_networkx_graph().nodes():
            current_properties = self.get_entity_info(entity)
            updated_properties = {**current_properties, **new_properties}
            self.graph.add_node(entity, **updated_properties)
            self.vector_store.add_texts([f"{entity}: {str(updated_properties)}"])
            return f"Entity '{entity}' updated successfully."
        else:
            return f"Entity '{entity}' not found in the graph."

    def delete_entity(self, entity: str):
        if entity in self.graph.get_networkx_graph().nodes():
            self.graph.get_networkx_graph().remove_node(entity)
            return f"Entity '{entity}' deleted from the graph."
        else:
            return f"Entity '{entity}' not found in the graph."

    def get_all_entities(self) -> List[str]:
        return list(self.graph.get_networkx_graph().nodes())

    def get_all_relationships(self) -> List[Tuple[str, str, str]]:
        return [(u, v, d.get('relationship', '')) for u, v, d in self.graph.get_networkx_graph().edges(data=True)]