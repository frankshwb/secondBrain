import os
import sys
import re
import chromadb
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder

# enable config import
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config


# -------------------------
# LOAD MODELS ONCE (IMPORTANT)
# -------------------------
embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
reranker = CrossEncoder(config.RERANKING_MODEL)

client = chromadb.PersistentClient(path=config.CHROMA_PATH)
collection = client.get_collection("second_brain")


# -------------------------
# KEYWORD EXTRACTION (LIGHTWEIGHT)
# -------------------------
def extract_keywords(text):

    words = re.findall(r"[A-Za-z0-9\-]{3,}", text.lower())

    stopwords = {
        "the", "and", "for", "with", "what", "does",
        "have", "about", "this", "that", "you", "are",
        "was", "from", "into", "can", "how"
    }

    keywords = [w for w in words if w not in stopwords]

    return keywords[:10]


# -------------------------
# HYBRID RETRIEVAL MERGE
# -------------------------
def merge_results(vec_results, kw_results):

    seen = set()
    merged_docs = []
    merged_meta = []

    def add_results(results):

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        for doc, meta in zip(docs, metas):

            source = meta.get("source", "unknown") if meta else "unknown"
            key = source + doc[:50]

            if key not in seen:
                seen.add(key)
                merged_docs.append(doc)
                merged_meta.append(meta)

    add_results(vec_results)
    add_results(kw_results)

    return merged_docs, merged_meta

# -------------------------
# RERANK FUNCTION
# -------------------------
def rerank(question, docs, metas, top_k=4):

    pairs = [(question, doc) for doc in docs]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(docs, metas, scores),
        key=lambda x: x[2],
        reverse=True
    )

    top = ranked[:top_k]

    reranked_docs = [t[0] for t in top]
    reranked_meta = [t[1] for t in top]

    return reranked_docs, reranked_meta


# -------------------------
# MAIN ASK FUNCTION
# -------------------------
def ask(question):

    query_embedding = embed_model.encode([question])[0]

    keywords = extract_keywords(question)
    keyword_query = " ".join(keywords)

    # VECTOR SEARCH
    vector_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3,
        include=["documents", "metadatas"]
    )

    # KEYWORD SEARCH
    keyword_results = collection.query(
        query_texts=[keyword_query],
        n_results=3,
        include=["documents", "metadatas"]
    )

    # MERGE
    merged_docs, merged_meta = merge_results(vector_results, keyword_results)
    
    #RERANK
    context_chunks, metadatas = rerank(question, merged_docs, merged_meta)

    # -------------------------
    # DEBUG OUTPUT
    # -------------------------
    print("\n--- Retrieved Context (Hybrid) ---\n")

    sources = []

    for i, (chunk, meta) in enumerate(zip(context_chunks, metadatas)):

        source = meta.get("source", "unknown") if meta else "unknown"
        doc_type = meta.get("type", "unknown") if meta else "unknown"

        sources.append(source)

        print(f"Chunk {i+1}")
        print(f"Source: {source}")
        print(f"Type: {doc_type}")
        print("\nText:")
        print(chunk[:400])
        print("\n--------------------------\n")


    # -------------------------
    # PROMPT
    # -------------------------
    context = "\n\n".join(context_chunks)

    prompt = f"""
        You are an AI assistant using a private knowledge base.

        Use ONLY the provided context to answer.

        If you are unsure, say so.

        Context:
        {context}

        Question:
        {question}

        After your answer, list sources used.

        Sources:
        {sources}

        Answer:
        """


    # -------------------------
    # OLLAMA CALL
    # -------------------------
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": config.LLM_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]


# -------------------------
# CHAT LOOP
# -------------------------
def chat():

    print("\nAI Knowledge Brain ready (Hybrid RAG + Sources).\n")

    while True:

        question = input("Ask: ")

        if question.lower() in ["exit", "quit"]:
            break

        answer = ask(question)

        print("\nAnswer:\n")
        print(answer)


if __name__ == "__main__":
    chat()