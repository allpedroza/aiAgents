import os
import json
import logging
import re
import time
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Optional
import numpy as np
from datetime import datetime

# Table utilities
def parse_table_name(table_text: str) -> str:
    """Extract table name from text"""
    patterns = [
        r'([a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)', # project.dataset.table
        r'([a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)',                 # dataset.table
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, table_text)
        if matches:
            return matches[0]
    
    return ""

def extract_tables_from_txt(file_path: str) -> List[str]:
    """Extract table names from a file"""
    if not os.path.exists(file_path):
        print(f"Arquivo não encontrado: {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find table names in BigQuery format
    table_pattern = r'([a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)'
    tables = re.findall(table_pattern, content)
    
    # Deduplicate
    return list(set(tables))

class CheckResults:
    def __init__(self, RESULTS_FILE):
        self.RESULTS_FILE = RESULTS_FILE
        
    def check_resultados_file(self):
        """Check if resultados_5.txt exists and create a sample if not found"""
        if not os.path.exists(self.RESULTS_FILE):
            print(f"Arquivo resultados_5.txt não encontrado em: {self.RESULTS_FILE}")
            print("Criando arquivo de exemplo...")

            sample_content = """
            tim-bigdata-prod-e305.trusted_ctp.dw_r_ctp_rg_hora
            Tabela que armazena dados de consumo de dados por rating group por hora.
            Informações sobre tráfego, incluindo volumes de download e upload em redes 3G, 4G e 5G.

            tim-bigdata-prod-e305.trusted_ctp.dw_r_ctp_user_detail
            Tabela com informações detalhadas de consumo por usuário.
            Inclui dados demográficos, tipo de plano, e detalhes de uso.

            tim-bigdata-prod-e305.curated_ctp.kpi_user_consumption
            Tabela agregada de KPIs de consumo por usuário.
            Métricas agregadas por diferentes dimensões.
            """
            os.makedirs(os.path.dirname(self.RESULTS_FILE), exist_ok=True)

            with open(self.RESULTS_FILE, 'w', encoding='utf-8') as f:
                f.write(sample_content)

            print(f"Arquivo de exemplo criado em: {self.RESULTS_FILE}")
            print("Você deve substituí-lo com seu arquivo real de mapeamento de tabelas.")

class JsonFormatter(logging.Formatter):
    def format(self, record):
        # Garante que o conteúdo do log seja um dicionário
        if isinstance(record.msg, dict):
            log_record = record.msg.copy()  # Evita modificar o original
        else:
            log_record = {"message": record.getMessage()}
        
        # Adiciona o nível de severidade
        log_record["severity"] = record.levelname
        return json.dumps(log_record, ensure_ascii=False)

# Classe que encapsula o logger e a função de logging
class QueryExecutorLogger:
    def __init__(self, LOGS_DIR, MODEL_NAME: str,USER_NAME: str):
        self.MODEL_NAME = MODEL_NAME
        self.LOGS_DIR = LOGS_DIR
        self.USER_NAME = USER_NAME
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger("agent_logger")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # Handler para arquivo com append
        if not logger.handlers:
            log_file_path = os.path.join(self.LOGS_DIR, "execution.log")
            file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
            file_handler.setFormatter(JsonFormatter())
            logger.addHandler(file_handler)
                    
        return logger
    
    def map_metadata(self, message):
        sys_metadata = llm.generate([messages[0].content]).generations[0][0].generation_info['usage_metadata']
        human_metadata = llm.generate([messages[1].content]).generations[0][0].generation_info['usage_metadata']
        
        metadata = {
            "system_metadata" :sys_metadata,
            "human_metadata": human_metadata
        }
        
        return metadata 

    def check_stats(self, start_time: float, result: Dict[str, Any], query: str) -> Dict[str, Any]:
        execution_time = time.time() - start_time
        timestamp = datetime.now().isoformat()
              
        log_data = {
            "user_name": self.USER_NAME,
            "date": timestamp,
            "query": query,
            "query_length_chars": len(query),
            "discovered_table": result.get("discovered_tables", []),
            "validated_table": result.get("validated_tables", []),
            "invalid_table": result.get("invalid_tables", []),
            "discovered_table_count": len(result.get("discovered_tables", [])),
            "validated_table_count": len(result.get("validated_tables", [])),
            "invalid_table_count": len(result.get("invalid_tables", [])),
            "discoverd_metadata_usage": result["discovery_metadata"],
            "response_metadata_usage": result["response_metadata"],
            "latency_ms": round(execution_time * 1000),
            "llm_model": self.MODEL_NAME,
            "response": result["response"],
            "user_feedback": result["user_feedback"],
            "user_justification": result["user_justification"],
        }

        self.logger.info(log_data)   
        return log_data
