import re
import json
import time
import logging

from pydantic import BaseModel
from typing import Optional, List
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from app.config import llm, METADATA_FILE, PARTITION_FILE
from app.utils.utils import partition_info, combine_infos_key, print_menu, query_cost_mb

## STATE

class QueryAssistantState(BaseModel):
    user_need: Optional[str] = None
    option: Optional[str] = None
    option_metrics: Optional[str] = None
    table_name: Optional[List[str]] = None
    table_metadata: Optional[List[str]] = None
    candidate_tables: Optional[List[str]] = None
    selected_table: Optional[str] = None
    user_query: Optional[str] = None
    generated_query: Optional[str] = None
    be_optimized: Optional[bool] = None
    other_optimizations: Optional[str] = None
    optimized_query: Optional[str] = None
    user_query_mb: Optional[float] = None
    optimized_query_mb: Optional[float] = None
    savings_mb: Optional[float] = None
    error_message: Optional[str] = None
    response: Optional[str] = None

### GRAPH ###

def build_graph():
    """Constrói e compila o grafo LangGraph.

    Returns:
        A aplicação LangGraph compilada.
    """
    # Define os nós
    graph = StateGraph(QueryAssistantState)

    graph.add_node("validate_table_existence", validate_table_existence)
    graph.add_node("table_metadata_capture", table_metadata_capture)
    graph.add_node("call_dda_agent", call_dda_agent)
    graph.add_node("validate_user_need", validate_user_need)
    graph.add_node("bq_writer", bq_writer)
    graph.add_node("query_judge", query_judge)
    graph.add_node("query_optimizer", query_optimizer)
    graph.add_node("optimization_metrics", optimization_metrics)
    graph.add_node("query_response", query_response)
    graph.add_node("end", lambda state: state)

    
    def decide_entry(state:QueryAssistantState):
        """ Define o nó de entrada com base no input do usuario."""
        option = state.option
        if option == '1':
            return "validate_table_existence"
        elif option == '2':
            #return "call_dda_agent"
            return "end"
        elif option == '3':
            return "table_metadata_capture"
        else:
            logging.warning("[Graph] Opção inválida. Encerrando execução.1")
            return "end"
        
    def decide_optimization(state:QueryAssistantState):
        """ Define o nó após a avaliação sobre otimização"""
        be_optimized = state.be_optimized
        if be_optimized == True:
            return "query_optimizer"
        if be_optimized == False:
            return "end"
        else:
            logging.warning("[Graph] Opção inválida. Encerrando execução.2")
            return "end"
        
    def decide_metrics(state:QueryAssistantState):
        """ Define se as métricas de otimização serão construídas."""
        option = state.option_metrics
        if option == '1':
            return "optimization_metrics"
        elif option == '2':
            return "end"
        else:
            #logging.warning("[Graph] Opção inválida. Encerrando execução.3")
            return "end"
        
    graph.set_conditional_entry_point(decide_entry)
    graph.add_edge("validate_table_existence", "validate_user_need")
    graph.add_edge("call_dda_agent", "validate_user_need")
    graph.add_edge("validate_user_need", "bq_writer")
    graph.add_edge("table_metadata_capture", "query_judge")
    graph.add_edge("bq_writer", "query_judge")
    graph.add_conditional_edges("query_judge", decide_optimization, {
        "query_optimizer": "query_optimizer",
        "end":"end"})
    graph.add_conditional_edges("query_optimizer", decide_metrics, {
        "optimization_metrics": "optimization_metrics",
        "end":"end"})

    # Compila uma vez
    compiled_graph = graph.compile()

    # Gera o Mermaid a partir do mesmo grafo compilado
    mermaid_code = compiled_graph.get_graph().draw_mermaid()
    with open("query_graph.mmd", "w") as f:
        f.write(mermaid_code)

    return compiled_graph

### NODES ###
def table_metadata_capture(state: QueryAssistantState) -> QueryAssistantState:
    """Nó: Captura nome das tabelas com base na query do usuário e carrega os metadados."""    
    return state

# def validate_table_existence(state: QueryAssistantState) -> QueryAssistantState:
#     """Nó: Verifica se a tabela fornecida existe nos metadados carregados."""
#     logging.info(f"[Node: validate_table_existence] Verificando existência da tabela: {state.table_name}")
#     with open(METADATA_FILE, "r") as f:
#         metadata = json.load(f)
   
#     table_name_to_find = state.table_name
#     found_table = None
#     if table_name_to_find in metadata:
#         logging.info(f"[Node: validate_table_existence] Tabela '{table_name_to_find}' encontrada.")
#         print("Tabela Encontrada")

#         found_table_metadata = {
#             "table_id": table_name_to_find,
#             "columns": metadata[table_name_to_find]
#         }

#         state.table_metadata = found_table_metadata  # agora é um dict válido

#         logging.info(f"[Node: validate_table_existence] Verificando existência dos dados de particionamento: {state.table_name}")
#         partition = partition_info(table_name_to_find, PARTITION_FILE)
#         print(f'partition: {partition}')
#         if partition:
#             combined = combine_infos_key(found_table_metadata, partition)
#         else:
#             combined = found_table_metadata

#         state.table_metadata = combined
#         print(f"COMB. {combined}")
#         state.error_message = None
#     else:
#         logging.warning(f"[Node: validate_table_existence] Tabela '{table_name_to_find}' não encontrada nos metadados.")
#         error_msg = f"A tabela '{table_name_to_find}' não foi encontrada nos metadados disponíveis em {METADATA_FILE}."
#         state.table_metadata = None
#         state.error_message = error_msg
#         state.response = error_msg
        
#     return state

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
            #print(f'Partição: {partition}')
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
    
    ## SE OS METADADOS DA TABELA FOREM VAZIO >> CARREGAR >> CASO DO DDA
    prompt = f""" 
    A partir da seguinte necessidade do usuario e metadados sobre a tabela, informe se você 
    """
    return state

def call_dda_agent(state: QueryAssistantState) -> QueryAssistantState:
    """Nó: Chamar o modulo do DDA para fornecer as tabelas em potencial para necessidade do usuario"""
    return state

# def bq_writer(state: QueryAssistantState) -> QueryAssistantState:
#     """Nó: Escreve query otimizada para o BigQueryy com base numa tabela e necessidade do usuario de uma empresa de telecomunicações"""
    
#     table = state.table_name if state.table_exists else state.selected_table
#     metadata = state.table_metadata
#     schema_str = metadata.get("columns")
#     partition_info = metadata.get('partitioning') or 'Sem particionamento'
#     clustering_info = metadata.get('clustering') or 'Sem clustering'
    
#     messages = [
#         SystemMessage(content = f"""Você é um especialista em consultas para BigQuery."""),
#         HumanMessage(content = f""" 
#         Necessidade do usuário: {state.user_need}
#         Tabela selecionada: {table}
#         Schema da tabela: {schema_str}
#         Particionamento: {partition_info}
#         Clustering: {clustering_info}

#         Forneça somente a query para o BigQuery utilizando a tabela e informações fornecidas.
#         """)]
    
#     response = llm.invoke(messages)
#     print(f'bq_writer:{response}')
#     state.generated_query = response.strip()
#     return state

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
    #print(f"[bq_writer] Query final gerada:\n{response}")
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
    #print(f'query_judge: {response}')
    # state.other_optimizations = response
    # if 'TRUE' in response.upper():
    #     state.be_optimized = True
    # elif 'FALSE' in response.upper():
    #     state.be_optimized = False
    # else:
    #     logging.warning(f"[Node: query_judge] Não há um veredito sobre a query.")
    # return state
    
    # Divide a resposta no ponto onde aparece "Query Otimizável:"
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
    
    #TODO: Query otimizada deve estar em alinhamento com sugestões do judge.
    
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
    #print(f'query_optimizer: {response}')
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
    
    #print(f'Otimização proporcionada: {otimizacao} MB')
    return state

def query_response(state: QueryAssistantState) -> QueryAssistantState:
    """Nó: Gera a query final para o usuario"""
    
    return state

### MAIN ###
def init_state(option:str, table_name: str| None = None, user_need: str | None = None, user_query: str | None = None) -> QueryAssistantState:
    """Initialize graph state"""
    return QueryAssistantState(
        user_need = user_need,
        option = option,
        option_metrics = None,
        table_name = table_name,
        table_exists = None,
        table_metadata = None,
        candidate_tables = None,
        selected_table = None,
        user_query = user_query,
        generated_query = None,
        be_optimized = None,
        other_optimizations = None,
        optimized_query = None,
        user_query_mb = None,
        optimized_query_mb = None,
        savings_mb = None,
        error_message = None,
        response = None)

def run_agent(run_mode = 'menu',user_query=None):
    logging.info("Construindo o grafo do Query Assistant")
    try:
        graph = build_graph()
        logging.info("Grafo construído com sucesso.")
        print("Grafo de processamento pronto.")
    except Exception as e:
        logging.exception("Erro crítico ao construir o grafo.")
        print(f"Erro crítico ao construir o grafo: {e}. Encerrando.")
        sys.exit(1)
    
    table_name = None
    user_need = None

    while True:
        start_time = time.time()
        if run_mode == 'menu':
            print_menu()
            option = input("Escolha uma opção: ").strip()

            if option == '1':
                raw_input = input("Para quais tabelas você deseja fazer a query? ").strip()
                table_name = [t.strip() for t in raw_input.split(",") if t.strip()]
                user_need = input("Descreva de forma completa a sua necessidade: ").strip()

            if option == '2':
                #user_need = input("Descreva de forma completa a sua necessidade: ").strip()
                print(f"Opção ainda não disponível.")
                print("\nEncerrando o sistema. Obrigado!")
                break

            if option == '3':
                user_query = input("Forneça a sua consulta do BigQuery: ").strip()
                
            elif option == "0":
                logging.info("Usuário solicitou encerramento.")
                print("\nEncerrando o sistema. Obrigado!")
                break
        elif run_mode == 'auto':
            user_query = user_query
            option_metrics = '1'
            

        try:
            init_run_state = init_state(option=option,
                                        table_name=table_name,
                                        user_need=user_need,
                                        user_query=user_query,
                                       )
            logging.debug(f"Estado inicial para o grafo: {init_run_state}")
            result_state = graph.invoke(init_run_state)
            
            print("-"*35 + ' RESPOSTA DO QUERY ASSISTANT '+ "-"*35)
            print(f'Query Otimizada:\n{result_state['optimized_query']}')
            
            if result_state['option_metrics'] == '1':
                print(f"Estimativa de processamento para query original: {result_state['user_query_mb']} MB")
                print(f"Estimativa de processamento para query otimizada: {result_state['optimized_query_mb']} MB")
                print(f"Estimativa de otimização: {result_state['savings_mb']} MB")
            print(f'\nInformações sobre otimização:\n{result_state['other_optimizations']}')
            
            end_time = time.time()
            logging.info(f"Grafo executado em {end_time - start_time:.2f} segundos")
            if result_state and isinstance(result_state, dict):
                response_message = result_state.get("response", "Nenhuma resposta textual gerada.")
                
            else:
                logging.error(f"Resultado inesperado ou inválido do grafo: {result_state}")
                print("Ocorreu um erro inesperado durante o processamento. Verifique os logs.")
    
        except Exception as e:
            logging.exception(f"Erro durante a execução do grafo ou processamento")
            print(f"\nErro inesperado ao processar a solicitação: {e}")



run_agent()
