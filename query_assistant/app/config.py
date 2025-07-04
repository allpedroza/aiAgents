import os
import logging
from langchain_google_vertexai import VertexAI


# Configure MODELs INFO
PROJECT_ID = 'tim-sdbx-datagovernance-c463'
LOCATION = 'us-east1'
MODEL_NAME = "gemini-2.0-flash"
MODEL_LIGHT_NAME = "gemini-1.5-flash"

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)

# Configure the data paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUTS_DIR = os.path.join(DATA_DIR, "input")
RESULTS_FILE = os.path.join(DATA_DIR, "results")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
#METADATA_FILE = os.path.join(DATA_DIR, "metadata.json") 
METADATA_FILE = os.path.join(DATA_DIR, "metadata_colunas.json") 
PARTITION_FILE = os.path.join(CACHE_DIR, "partition.json")

# Ensure directories exist
os.makedirs(os.path.join(DATA_DIR, "extract"), exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def load_config():  
    return {
        "PROJECT_ID":PROJECT_ID,
        "LOCATION": LOCATION,
        "MODEL_NAME":MODEL_NAME,
        "MODEL_LIGHT_NAME": MODEL_LIGHT_NAME,
        "INPUTS_DIR": INPUTS_DIR,
        "RESULTS_FILE": RESULTS_FILE,
        "DATA_DIR": DATA_DIR,
        "CACHE_DIR": CACHE_DIR,
        "LOGS_DIR": LOGS_DIR,
        "METADATA_FILE":METADATA_FILE,
        "PARTITION_FILE":PARTITION_FILE,
    }

llm = None
try:
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or PROJECT_ID != "seu-projeto-gcp":
        llm = VertexAI(model_name=MODEL_NAME,project=PROJECT_ID, location=LOCATION, temperature=0)
        logging.info(f"Vertex AI LLM ({MODEL_NAME}) inicializado com sucesso para projeto {PROJECT_ID} em {LOCATION}.")
    else:
        logging.warning("Credenciais Google Cloud ou PROJECT_ID não configurados via variáveis de ambiente ou valores \
        padrão não alterados. LLM não inicializado.")
except ImportError:
    logging.error("Erro ao importar langchain_google_vertexai. Verifique a instalação.")
except Exception as e:
    logging.error(f"Erro ao inicializar Vertex AI LLM: {e}. Verifique suas credenciais e configurações.")
