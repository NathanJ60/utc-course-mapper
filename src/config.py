from pathlib import Path

# Chemins
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
CATALOGUE_DIR = ROOT_DIR / "catalogue-uv"

# Fichiers
UV_CATALOGUE_PDF = CATALOGUE_DIR / "uv_catalogue.pdf"
UV_PARSED_JSON = DATA_DIR / "uv_parsed.json"
UV_EMBEDDINGS_JSON = DATA_DIR / "uv_embeddings.json"
QDRANT_DB_PATH = DATA_DIR / "qdrant_db"

# OpenAI
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072

# Qdrant
COLLECTION_NAME = "uv_utc"
