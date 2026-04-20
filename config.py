from dotenv import load_dotenv
import os

# .env Datei laden
load_dotenv()

VAULT_PATH = os.getenv("VAULT_PATH")
PDF_PATH = os.path.join(VAULT_PATH, "05_resources/pdf")
WIKI_PATH = os.path.join(VAULT_PATH, "90_llm_wiki")
WIKI_PAGES_PATH = os.path.join(WIKI_PATH, "pages")
WIKI_SOURCES_PATH = os.path.join(WIKI_PATH, "sources")
WIKI_SYSTEM_PATH = os.path.join(WIKI_PATH, "system")
WIKI_INDEX_FILE = os.path.join(WIKI_SYSTEM_PATH, "index.md")
WIKI_LOG_FILE = os.path.join(WIKI_SYSTEM_PATH, "log.md")
WIKI_SCHEMA_FILE = os.path.join(WIKI_SYSTEM_PATH, "schema.md")

# Chroma DB Speicher
CHROMA_PATH = os.getenv("CHROMA_PATH")

# Embedding Model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

#Reranking Model
RERANKING_MODEL = os.getenv("RERANKING_MODEL")

# Ollama Modell
LLM_MODEL = os.getenv("LLM_MODEL")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")