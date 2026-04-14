import os

# Pfad zu deinem Obsidian Vault
VAULT_PATH = "/Users/frsc4/SecondBrain"

PDF_PATH = os.path.join(VAULT_PATH, "05_resources/pdf")

# Chroma DB Speicher
CHROMA_PATH = "./data/chroma_db"

# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

#Reranking Model
RERANKING_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Ollama Modell
LLM_MODEL = "llama3"