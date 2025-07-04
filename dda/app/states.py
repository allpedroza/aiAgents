from typing import TypedDict, List, Dict, Any, Optional, Literal

class GraphState(TypedDict):
    query: str
    discovered_tables: List[Dict[str, Any]]
    validated_tables: List[Dict[str, Any]]
    invalid_tables: List[Dict[str, Any]]
    discovery_metadata: List[Dict[str, Any]]
    response_metadata: List[Dict[str, Any]]
    response: Optional[str]
    user_feedback: Optional[Literal["positiva", "negativa"]]
    user_justification: Optional[str]

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
        "response": None,
        "user_feedback": None,
        "user_justification": None,
    }
