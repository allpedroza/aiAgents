import re
import json
import time
import logging
from tim_ai_agents.query_assistant.app.agent.graph import build_graph
from tim_ai_agents.query_assistant.app.agent.state import QueryAssistantState, init_state
from tim_ai_agents.query_assistant.app.utils.utils import partition_info, combine_infos_key, print_menu, query_cost_mb


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
