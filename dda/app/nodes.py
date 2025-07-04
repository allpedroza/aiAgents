import re
import json
from google import genai
from langchain_google_vertexai import VertexAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, List, Dict, Any, Optional, TypedDict, Union, Literal, Tuple
from tim_ai_agents.data_discovery.app.config import load_config
from tim_ai_agents.data_discovery.app.agent.state import GraphState
from tim_ai_agents.data_discovery.app.retriever.tools import GeminiDocumentProcessor
from tim_ai_agents.data_discovery.app.utils.utils import extract_tables_from_txt

config = load_config()
MODEL_NAME = config["MODEL_NAME"]
MODEL_LIGHT_NAME = config["MODEL_LIGHT_NAME"] 
CACHE_DIR = config["CACHE_DIR"] 
PROJECT_ID = config["PROJECT_ID"] 
LOCATION = config["LOCATION"] 
RESULTS_FILE = config["RESULTS_FILE"]

class DataDiscoveryAgent:
    def __init__(self):
        """Initialize the data discovery agent with the LLM"""
        self.llm = VertexAI(model_name=MODEL_NAME,project=PROJECT_ID, location=LOCATION, temperature=0)
        self.processor = GeminiDocumentProcessor.get_instance(MODEL_NAME=MODEL_NAME, CACHE_DIR=CACHE_DIR)
    
    def discover_tables(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Discover tables relevant to the query"""
        # Get processor instance
        processor = GeminiDocumentProcessor.get_instance(MODEL_NAME=MODEL_NAME, CACHE_DIR=CACHE_DIR)
        
        # Query document using Dartboard ranking
        relevant_chunks = processor.query_document(query, top_k)
        
        if not relevant_chunks:
            return []
        
        # Format context
        context_text = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
        # print(f'CONTEXTO: {context_text} --- {len(context_text)}')
        
        # Create prompt
        system_prompt = """You are a data discovery assistant for a telecom company. 
        Your task is to identify the most relevant tables for the user's query.
        Analyze the context provided and find tables that contain data needed to answer the query.
        Format your response as a JSON array with objects containing these fields:
        - table_name: The full name of the table (project.dataset.table)
        - description: A brief description of what this table contains
        - relevance: Why this table is relevant to the query
        - confidence: Your confidence level (high, medium, low) that this table is useful

        Only include tables that are specifically mentioned in the context. Do not make up table names."""
                
        human_prompt = f"""Query: {query}

        Context:
        {context_text}

        Identify and list the tables from the context that would be most relevant to answer this query. 
        Return your response as a valid JSON array."""
        
        # Create messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        # Get response
        try:
            response = self.llm.invoke(messages)
            sys_metadata = self.llm.generate([messages[0].content]).generations[0]\
            [0].generation_info['usage_metadata']
            human_metadata = self.llm.generate([messages[1].content]).generations[0]\
            [0].generation_info['usage_metadata']
            
            discovery_metadata = [{
                "sys_metadata": sys_metadata,
                "human_metadata": human_metadata
            }]
            
            # Extract JSON from response
            content = response
            json_match = re.search(r'(\[\s*\{.*\}\s*\])', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            
            # Parse JSON
            discovered_tables = json.loads(content)
            return discovered_tables, discovery_metadata
        except Exception as e:
            print(f"Erro durante a descoberta de tabelas: {e}")
            return []

class TableValidatorAgent:
    def __init__(self):
        """Initialize the table validator agent"""
        self.llm = VertexAI(model_name=MODEL_LIGHT_NAME,project=PROJECT_ID, location=LOCATION, temperature=0)
        
    def validate_tables(self, query: str, discovered_tables: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Validate tables against the database"""
        # Extract available tables from resultados_5.txt
        available_tables = set(extract_tables_from_txt(RESULTS_FILE))
        
        valid_tables = []
        invalid_tables = []
        
        for table in discovered_tables:
            table_name = table.get("table_name", "")
            if not table_name:
                # Try to extract from description
                table_name = parse_table_name(table.get("description", ""))
                if not table_name:
                    invalid_tables.append({**table, "reason": "Nome de tabela ausente ou inválido"})
                    continue
            
            # Check if table exists
            if table_name in available_tables:
                valid_tables.append(table)
            else:
                # Try partial matches
                found_match = False
                for available_table in available_tables:
                    if table_name in available_table or available_table.endswith(table_name):
                        # Update with full name
                        updated_table = {**table, "table_name": available_table}
                        valid_tables.append(updated_table)
                        found_match = True
                        break
                
                if not found_match:
                    invalid_tables.append({**table, "reason": "Tabela não encontrada no banco de dados"})
        
        return valid_tables, invalid_tables

class ResponseGenerator:
    def __init__(self):
        """Initialize the response generator"""
        self.llm = VertexAI(model_name=MODEL_LIGHT_NAME,project=PROJECT_ID, location=LOCATION, temperature=0)
    
    def generate_response(self, query: str, validated_tables: List[Dict[str, Any]], invalid_tables: List[Dict[str, Any]]) -> str:
        """Generate formatted response"""
        if not validated_tables:
            return "Não foram encontradas tabelas relevantes para sua consulta nos dados disponíveis."
        
        system_prompt = """Você é um Assistente AI especializado em análise de dados para uma empresa de telecomunicações brasileira.
        Sua tarefa é responder sobre a query do usuário baseada nas tabelas validadas fornecidas.

        Formate cada recomendação de tabela exatamente neste formato:

        - [nome_completo_da_tabela] (Ex: tim-bigdata-prod-e305.trusted_ctp.dw_r_ctp_rg_hora)
        - Descrição: Uma breve descrição da tabela e seu conteúdo.
        - Informações relevantes para a query: Explique quais colunas e informações específicas nesta tabela ajudam a responder à query do usuário.

        Use português brasileiro na sua resposta.
        Seja conciso mas informativo.
        Inclua apenas informações mencionadas nas tabelas validadas.
        Não invente informações ou colunas."""
        
        # Convert tables to string
        tables_content = json.dumps(validated_tables, indent=2, ensure_ascii=False)
        # print(f'TABLES CONTENT: {tables_content} --- {len(tables_content)}')
        human_prompt = f"""Query: {query}

        Tabelas validadas:
        {tables_content}
        

        Gere uma resposta formatada em português brasileiro que recomende estas tabelas para responder à query."""
        
        # Create messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        # Get response
        try:
            response = self.llm.invoke(messages)
            sys_metadata = self.llm.generate([messages[0].content]).generations[0]\
            [0].generation_info['usage_metadata']
            human_metadata = self.llm.generate([messages[1].content]).generations[0]\
            [0].generation_info['usage_metadata']
            
            response_metadata = [{
                "sys_metadata": sys_metadata,
                "human_metadata": human_metadata
            }]            
            return response, response_metadata
        except Exception as e:
            print(f"Erro ao gerar a resposta: {e}")
            return "Erro ao gerar a resposta. Por favor, tente novamente."
    
    def user_feedback (self) -> str:
                        # Show response
        print("\n--- FEEDBACK DO USUARIO ---")
        print("Qual o seu nível de satisfação com a resposta?")
        
        while True:
            print("1 - CURTI 👍")
            print("2 - NÃO CURTI 👎")

            choice = input("Escolha uma opção: ").strip()
            if choice in ("1", "2"):
                break
            print("Entrada inválida. Por favor, digite 1 ou 2.")
            

        feedback_map = {"1": "positiva", "2": "negativa"}
        feedback = feedback_map.get(choice, "positiva")

        if feedback == "negativa":
            justification = input("Por favor, descreva o motivo da sua insatisfação: ").strip()

        else:
            justification = []
        return feedback, justification

# LangGraph implementation
def init_state(query: str) -> GraphState:
    """Initialize graph state"""
    return {
        "query": query,
        "discovered_tables": [],
        "validated_tables": [],
        "invalid_tables": [],
        "discovery_metadata": [],
        "response_metadata": [],
        "response": None
    }

def discover_tables_node(state: GraphState) -> GraphState:
    """Discover tables node"""
    # Create discovery agent
    discovery_agent = DataDiscoveryAgent()
    
    # Discover tables
    discovered_tables,discovery_metadata = discovery_agent.discover_tables(state["query"])

    return {
        **state,
        "discovered_tables": discovered_tables,
        "discovery_metadata": discovery_metadata
    }

def validate_tables_node(state: GraphState) -> GraphState:
    """Validate tables node"""
    # Create validator agent
    validator_agent = TableValidatorAgent()
    
    # Validate tables
    validated_tables, invalid_tables = validator_agent.validate_tables(
        state["query"], 
        state["discovered_tables"],
    )
    
    return {
        **state,
        "validated_tables": validated_tables,
        "invalid_tables": invalid_tables
    }

def generate_response_node(state: GraphState) -> GraphState:
    """Generate response node"""
    # Create response generator
    response_generator = ResponseGenerator()
    
    # Generate response
    response,response_metadata = response_generator.generate_response(
        state["query"],
        state["validated_tables"],
        state["invalid_tables"],
    )
    
    # Show response
    print("\n--- RESPOSTA DO TIM DATA DISCOVERY ---")
    print(response)
    
    return {
        **state,
        "response": response,
        "response_metadata": response_metadata
    }

def user_feedback_node(state: GraphState) -> GraphState:
    """Generate feedback node"""
    # Create response generator
    response_generator = ResponseGenerator()

    # Feedback
    feedback, justification = response_generator.user_feedback()
        
    return {
        **state,
        "user_feedback": feedback,
        "user_justification": justification
    }

def should_end(state: GraphState) -> Literal["end"]:
    """Always end after generating response"""
    return "end"
