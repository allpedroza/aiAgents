from langgraph.graph import END, StateGraph, START

def build_graph():
    """Build and return the compiled graph"""
    
    from .nodes import discover_tables_node, validate_tables_node, generate_response_node,user_feedback_node, should_end
    from .state import GraphState
    
    # Create graph
    graph = StateGraph(GraphState)
    
    # Add nodes
    graph.add_node("discover_tables", discover_tables_node)
    graph.add_node("validate_tables", validate_tables_node)
    graph.add_node("generate_response", generate_response_node)
    graph.add_node("collect_user_feedback", user_feedback_node)

    
    # Add edges
    graph.add_edge("discover_tables", "validate_tables")
    graph.add_edge("validate_tables", "generate_response")
    graph.add_edge("generate_response", "collect_user_feedback")
    graph.add_conditional_edges(
        "collect_user_feedback",
        should_end,
        {
            "end": END
        }
    )
    
    # Set entry point
    graph.set_entry_point("discover_tables")
    
    # Compile graph
    return graph.compile()
