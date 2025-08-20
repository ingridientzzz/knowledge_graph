import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# _get_data_path removed - now using only DBT artifacts from code_docs/

def _get_artifacts_path():
    """Auto-detect the correct path to dbt artifacts directory."""
    env_path = os.getenv("DBT_ARTIFACTS_PATH")
    if env_path:
        return env_path
    
    # Auto-detect based on current working directory
    current_dir = Path.cwd()
    
    # If running from backend/ directory
    if current_dir.name == "backend":
        return "../../code_docs"
    # If running from chatdbt-lmstudio/ directory  
    elif current_dir.name == "chatdbt-lmstudio":
        return "../code_docs"
    # Fallback
    else:
        return "../../code_docs"

class Config:
    """Configuration class for ChatDBT with LM Studio."""
    
    def __init__(self):
        # LM Studio configuration
        self.LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
        self.LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
        self.LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "openai/gpt-oss-20b")  # Can be set to any model loaded in LM Studio
        
        # Data configuration - using only DBT artifacts
        self.DBT_ARTIFACTS_PATH = _get_artifacts_path()
        
        # Server configuration
        self.HOST = os.getenv("HOST", "0.0.0.0")
        self.PORT = int(os.getenv("PORT", 8000))
        
        # Index storage
        self.INDEX_STORAGE_PATH = os.getenv("INDEX_STORAGE_PATH", "./storage")
        
        # LLM settings
        self.LLM_CHUNK_SIZE = int(os.getenv("LLM_CHUNK_SIZE", 8192))  # Increased for PropertyGraph
        self.LLM_CHUNK_OVERLAP = int(os.getenv("LLM_CHUNK_OVERLAP", 400))
        self.EMBEDDING_CHUNK_SIZE = int(os.getenv("EMBEDDING_CHUNK_SIZE", 1024))
        
        # Retrieval settings
        self.SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", 10))
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))
        self.MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2048))
        
        # Index type selection
        self.USE_PROPERTY_GRAPH = os.getenv("USE_PROPERTY_GRAPH", "true").lower() == "true"
