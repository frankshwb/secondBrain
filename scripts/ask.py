import os
import sys

# ermöglicht Import von config.py
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config
import chromadb
from sentence_transformers import SentenceTransformer
import requests


# Embedding Modell 1x nur laden
embed_model = SentenceTransformer(config.EMBEDDING_MODEL)

def ask(question):

    # Verbindung zu Chroma
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)

    collection = client.get_collection("second_brain")

    # Embedding der Frage erstellen
    query_embedding = embed_model.encode([question])[0]

    # Retrieval
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3,
        include=["documents", "metadatas"]
    )

    context_chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    print("\n--- Retrieved Context (with sources) ---\n")

    for i, (chunk, meta) in enumerate(zip(context_chunks, metadatas)):

        print(f"Chunk {i+1}")
        print(f"Source: {meta['source']}")
        print(f"Type: {meta['type']}")
        print("\nText:")
        print(chunk[:400])
        print("\n--------------------------\n")

    
    context = "\n\n".join(context_chunks)

    # Prompt bauen
    prompt = f"""
You are an AI assistant using the user's private knowledge base.

Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

    # Anfrage an Ollama
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": config.LLM_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]


def chat():

    print("\nAI Knowledge Brain ready.")

    while True:

        question = input("\nAsk: ")

        if question.lower() in ["exit", "quit"]:
            break

        answer = ask(question)

        print("\nAnswer:\n")
        print(answer)


if __name__ == "__main__":
    chat()