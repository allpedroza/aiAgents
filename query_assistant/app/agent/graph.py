from langgraph.graph import END, StateGraph, START
from tim_ai_agents.query_assistant.app.agent.state import QueryAssistantState
from tim_ai_agents.query_assistant.app.agent.nodes import validate_table_existence, table_metadata_capture, call_dda_agent, validate_user_need, bq_writer, query_judge, query_optimizer, optimization_metrics, query_response

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
