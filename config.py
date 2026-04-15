from dotenv import load_dotenv
import os

# .env Datei laden
load_dotenv()

VAULT_PATH = os.getenv("VAULT_PATH")
PDF_PATH = os.path.join(VAULT_PATH, "05_resources/pdf")

# Chroma DB Speicher
CHROMA_PATH = os.getenv("CHROMA_PATH")

# Embedding Model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

#Reranking Model
RERANKING_MODEL = os.getenv("RERANKING_MODEL")

# Ollama Modell
LLM_MODEL = os.getenv("LLM_MODEL")