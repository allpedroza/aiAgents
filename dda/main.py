import os
import sys
import time
import logging
from tim_ai_agents.data_discovery.app.agent.graph import build_graph
from tim_ai_agents.data_discovery.app.agent.state import init_state
from tim_ai_agents.data_discovery.app.utils.utils import CheckResults, QueryExecutorLogger
from tim_ai_agents.data_discovery.app.retriever.tools import GeminiDocumentProcessor
from tim_ai_agents.data_discovery.app.config import load_config
from typing import TypedDict, List, Dict, Any, Optional


def initialize_environment() -> Dict[str, Any]:
    """Inicializa o ambiente e retorna a configuração"""
    try:
        return load_config()
    except ValueError as e:
        print(f"Erro de configuração: {e}")
        sys.exit(1)

def print_menu():
    """Print menu options"""
    print("\n" + "="*50)
    print("TIM DATA DISCOVERY")
    print("="*50)
    print("1 - Faça uma pergunta ao Data Discovery")
    print("2 - Modo Debug (visualizar detalhes da execução)")
    print("0 - Encerrar")
    print("="*50)
    
def run_interactive(USER_NAME=None):
    """Run interactive menu"""
    
    config = initialize_environment()
    MODEL_NAME = config["MODEL_NAME"]
    LOGS_DIR = config["LOGS_DIR"]
    RESULTS_FILE = config["RESULTS_FILE"]
    CACHE_DIR = config["CACHE_DIR"]
        
    # Check for resultados file
    checker = CheckResults(RESULTS_FILE=RESULTS_FILE)
    checker.check_resultados_file()
    
    # Initialize vector database at startup (SINGLE INITIALIZATION)
    print("Inicializando o banco vetorial...")
    processor = GeminiDocumentProcessor.get_instance(MODEL_NAME=MODEL_NAME, CACHE_DIR=CACHE_DIR)
    
    # Initialize with documento if needed
    if processor.index.ntotal == 0:
        print(f"Carregando dados de {RESULTS_FILE}...")
        processor.index_document(RESULTS_FILE)
    
    print(f"Banco vetorial inicializado com {processor.index.ntotal} vetores!")
    
    # Build graph once
    graph = build_graph()
    
    while True:
        print_menu()
        opcao = input("Escolha uma opção: ").strip()
        
        if opcao == "1":
            query = input("\nDigite sua pergunta sobre dados: ").strip()
            
            if not query:
                print("Pergunta vazia. Por favor, tente novamente.")
                continue
                
            print("\nProcessando sua pergunta...")
            start_time = time.time()
            
            try:
                # Initialize state with the query
                state = init_state(query)
                
                # Execute the graph with minimal output
                with open(os.devnull, 'w') as devnull:
                    # Redirect stdout temporarily
                    original_stdout = os.dup(1)
                    os.dup2(devnull.fileno(), 1)
                    
                    # Run graph
                    result = graph.invoke(state)
                    
                    # Restore stdout
                    os.dup2(original_stdout, 1)

                logger = QueryExecutorLogger(MODEL_NAME=MODEL_NAME, LOGS_DIR = LOGS_DIR, USER_NAME = USER_NAME)
                logger.check_stats(start_time, result, query)
            
            except Exception as e:
                print(f"Erro ao processar a pergunta: {e}")

        elif opcao == "2":
            query = input("\nDigite sua pergunta para modo debug: ").strip()
            
            if not query:
                print("Pergunta vazia. Por favor, tente novamente.")
                continue
                
            print("\nExecutando em modo debug...")
            start_time = time.time()
            
            try:
                # Initialize state with the query
                state = init_state(query)
                
                # Execute the graph
                result = graph.invoke(state)
                
                # Show condensed debug info
                print("\n--- MODO DEBUG ---")
                
                # Display validated tables
                print("\nTABELAS VALIDADAS:")
                if result["validated_tables"]:
                    for i, table in enumerate(result["validated_tables"], 1):
                        print(f"{i}. {table.get('table_name', 'N/A')} - {table.get('confidence', '')} confidence")
                else:
                    print("Nenhuma tabela validada.")
                
                # Display rejected tables
                print("\nTABELAS REJEITADAS:")
                if result["invalid_tables"]:
                    for i, table in enumerate(result["invalid_tables"], 1):
                        name = table.get('table_name', 'Sem nome')
                        reason = table.get('reason', 'Sem motivo')
                        print(f"{i}. {name} - Motivo: {reason}")
                else:
                    print("Nenhuma tabela rejeitada.")
                                
                logger = QueryExecutorLogger(MODEL_NAME=MODEL_NAME, LOGS_DIR = LOGS_DIR, USER_NAME = USER_NAME)
                logger.check_stats(start_time, result, query)
                
                # Show execution time
                execution_time = time.time() - start_time
                print(f"\nTempo de execução: {execution_time:.2f} segundos")
                
                # Ask for full details
                show_full = input("\nDeseja ver detalhes completos? (s/n): ").strip().lower()
                if show_full == 's':
                    print("\nDETALHES COMPLETOS DAS TABELAS DESCOBERTAS:")
                    for i, table in enumerate(result["discovered_tables"], 1):
                        print(f"\nTabela {i}:")
                        for key, value in table.items():
                            print(f"  {key}: {value}")
                
            except Exception as e:
                print(f"Erro no modo debug: {e}")
                
        elif opcao == "0":
            print("\nEncerrando o sistema. Obrigado!")
            break
            
        else:
            print("\nOpção inválida. Por favor, escolha uma opção válida.")

# Run the interactive menu if executed directly
if __name__ == "__main__":
    run_interactive()
