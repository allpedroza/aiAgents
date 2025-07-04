from pydantic import BaseModel
from typing import Optional, List

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
