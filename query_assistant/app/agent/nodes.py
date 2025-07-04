import json
import re
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from tim_ai_agents.query_assistant.app.utils.utils import partition_info, combine_infos_key, print_menu, query_cost_mb
from tim_ai_agents.query_assistant.app.config import llm, METADATA_FILE, PARTITION_FILE
from tim_ai_agents.query_assistant.app.agent.state import QueryAssistantState

def table_metadata_capture(state: QueryAssistantState) -> QueryAssistantState:
    """Nó: Captura nome das tabelas com base na query do usuário e carrega os metadados."""    
    return state

def validate_table_existence(state: QueryAssistantState) -> QueryAssistantState:
    """Nó: Verifica se as tabelas fornecidas existem nos metadados carregados."""
    logging.info(f"[Node: validate_table_existence] Verificando existência das tabelas: {state.table_name}")
    
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)

    found_metadata = []
    not_found_tables = []

    for table_name in state.table_name:
        if table_name in metadata:
            logging.info(f"[Node: validate_table_existence] Tabela '{table_name}' encontrada.")
            print(f"Metadados da tabela {table_name} encontrados")

            table_metadata = {
                "table_id": table_name,
                "columns": metadata[table_name]
            }

            # Verifica dados de partição
            logging.info(f"[Node: validate_table_existence] Verificando partição para: {table_name}")
            partition = partition_info(table_name, PARTITION_FILE)
            if partition:
                table_metadata = combine_infos_key(table_metadata, partition)

            found_metadata.append(table_metadata)
        else:
            logging.warning(f"[Node: validate_table_existence] Tabela '{table_name}' não encontrada.")
            not_found_tables.append(table_name)

    state.table_metadata = found_metadata if found_metadata else None

    if not_found_tables:
        error_msg = (
            f"As seguintes tabelas não foram encontradas nos metadados em {METADATA_FILE}: "
            f"{', '.join(not_found_tables)}"
        )
        state.error_message = error_msg
        state.response = error_msg
        logging.warning(f"[Node: validate_table_existence] {error_msg}")
    else:
        state.error_message = None
        state.response = None

    return state


def validate_user_need(state: QueryAssistantState) -> QueryAssistantState:
    """Nó: Verifica se a necessidade informada pelo usuário atende às necessidades para a query"""
    
    return state

def call_dda_agent(state: QueryAssistantState) -> QueryAssistantState:
    """Nó: Chamar o modulo do DDA para fornecer as tabelas em potencial para necessidade do usuario"""
    return state

def bq_writer(state: QueryAssistantState) -> QueryAssistantState:
    """
    Nó: Gera uma única query otimizada para o BigQuery com base nas múltiplas tabelas e na necessidade do usuário.
    """
    tables = state.table_name
    metadatas = state.table_metadata

    
    if not tables or not metadatas or len(tables) != len(metadatas):
        state.generated_query = None
        state.error_message = "Erro ao gerar query: tabelas e metadados não estão alinhados ou estão ausentes."
        return state

    # Constrói a descrição consolidada de todas as tabelas
    table_descriptions = []
    for table, metadata in zip(tables, metadatas):
        schema_str = metadata.get("columns")
        partition_info = metadata.get("partitioning") or 'Sem particionamento'
        clustering_info = metadata.get("clustering") or 'Sem clustering'

        table_description = f"""
        Tabela: {table}
        Schema: {schema_str}
        Particionamento: {partition_info}
        Clustering: {clustering_info}
        """
        table_descriptions.append(table_description.strip())

    full_context = "\n\n".join(table_descriptions)

    messages = [
        SystemMessage(content="Você é um especialista em consultas SQL para o BigQuery."),
        HumanMessage(content=f""" 
        Necessidade do usuário: {state.user_need}

        As seguintes tabelas estão disponíveis, com seus respectivos schemas e informações técnicas:

        {full_context}

        Com base nas informações acima, gere uma única query SQL para o BigQuery que atenda à necessidade do usuário.
        Utilize joins, CTEs ou qualquer construção necessária.
        Apenas forneça a query final, sem explicações.
        """)
    ]

    response = llm.invoke(messages)
    state.generated_query = response.strip()
    return state


def query_judge(state: QueryAssistantState) -> QueryAssistantState:
    """Nó: Avalia se é possível otimizar uma query ou a tabela com base na query e necessidade do usuario (se disponivel)"""
        
    ## PS: Para personas de BUs não faz sentido propor **Materialized View**
    if state.generated_query:
        query = state.generated_query
    elif state.user_query:
        query = state.user_query
    else:
        logging.warning(f"[Node: query_judge] Não há query para ser otimizada.")
        
    messages = [
        SystemMessage(content = f"""Você é expert em avaliar se a query para o BigQuery fornecida poderá ser otimizada"""),
        HumanMessage(content = f""" 
        Query: {query}
        
        **Responda no formato:**
        Query Otimizável: "TRUE" caso a query possa ser otimizada e com "FALSE" caso não possa. Não adicione qualquer outra informação nessa parte da resposta além disso.
        Garanta que o objetivo da query original seja mantido.
        Após o veredito, avalie também se há melhorias que possam ser feitas na tabela, como sugerir a criação de particionamento para melhorar performance e reduzir custos.
        Não forneça a query otimizada.
        """)]
    
    response = llm.invoke(messages)
    match = re.search(r"Query Otimizável:\s*(TRUE|FALSE)", response, re.IGNORECASE)
    
    if match:
        verdict = match.group(1).upper()
        if verdict == "TRUE":
            state.be_optimized = True
        elif verdict == "FALSE":
            state.be_optimized = False

        # Captura tudo que vem após a linha do veredito
        split_response = re.split(r"Query Otimizável:\s*(TRUE|FALSE)", response, flags=re.IGNORECASE)
        if len(split_response) > 2:
            other_text = split_response[2].strip()
            state.other_optimizations = other_text
        else:
            state.other_optimizations = None
    else:
        logging.warning(f"[Node: query_judge] Não foi possível identificar 'Query Otimizável: TRUE/FALSE' na resposta.")
        state.be_optimized = None
        state.other_optimizations = response  # armazena tudo como fallback

    return state

def query_optimizer(state: QueryAssistantState) -> QueryAssistantState:
    """Nó: Otimizar a query do usuário"""
       
    table = state.table_name
    
    if state.generated_query:
        query = state.generated_query
    elif state.user_query:
        query = state.user_query
    else:
        logging.warning(f"[Node: query_optime] Não há query para ser otimizada.")
    
    otimizacoes = state.other_optimizations
    messages = [
        SystemMessage(content = f""" Você é especialista em otimizar consultas para o BigQuery."""),
        HumanMessage(content = f"""
        Forneça a versão otimizada da consulta: {query} para a tabela {table}, considerando as seguintes melhorias, se houverem: {otimizacoes}  
        Não adicione qualquer informação além da query. Não adicione markdowns.
       """),
    ]
    response = llm.invoke(messages)
    state.optimized_query = response
    
    if state.option == '3':
        metrics_avaliation = input("Você deseja gerar as métricas da otimização?\n 1. SIM\n 2. NÃO\n").strip()
        state.option_metrics = metrics_avaliation
    
    return state
    
def optimization_metrics(state: QueryAssistantState) -> QueryAssistantState:
    """Nó: Avalia os ganhos da otimizaçao da query"""
    
    query_antiga = state.user_query
    query_otimizada = state.optimized_query
    consumo_antigo = query_cost_mb(query_antiga, 'query_original')
    state.user_query_mb = consumo_antigo
    
    cleaned_query = re.sub(r"^```sql\s*|```$", "", query_otimizada.strip(), flags=re.IGNORECASE)
    consumo_otimizado = query_cost_mb(cleaned_query, 'query_otimizada')
    
    state.optimized_query_mb = consumo_otimizado
    otimizacao = consumo_antigo - consumo_otimizado
    state.savings_mb = otimizacao
    
    return state

def query_response(state: QueryAssistantState) -> QueryAssistantState:
    """Nó: Gera a query final para o usuario"""
    
    return state
