import os
import sys
import re
from pathlib import Path
import chromadb
import requests
import json
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


def load_wiki_context(question, max_items=2, max_chars=1400):
    wiki_root = Path(config.WIKI_PATH)
    candidate_dirs = [wiki_root / "sources", wiki_root / "pages"]

    question_tokens = set(extract_keywords(question))
    if not question_tokens:
        question_tokens = set(re.findall(r"[A-Za-z0-9\-]{3,}", question.lower()))

    scored = []

    for directory in candidate_dirs:
        if not directory.exists():
            continue

        for file_path in directory.glob("*.md"):
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception:
                continue

            if not content.strip():
                continue

            content_tokens = set(re.findall(r"[A-Za-z0-9\-]{3,}", content.lower()))
            overlap = len(question_tokens.intersection(content_tokens))

            if overlap == 0:
                continue

            rel_path = str(file_path.relative_to(wiki_root)).replace("\\", "/")
            scored.append((overlap, rel_path, content[:max_chars]))

    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[:max_items]

    docs = [item[2] for item in top]
    metas = [{"source": item[1], "type": "wiki"} for item in top]
    return docs, metas


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

    wiki_docs, wiki_meta = load_wiki_context(question, max_items=2)

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
    keyword_embedding = embed_model.encode(keyword_query if keyword_query else question)
    keyword_results = collection.query(
        query_embeddings=[keyword_embedding.tolist()],
        n_results=3,
        include=["documents", "metadatas"]
    )

    # MERGE
    merged_docs, merged_meta = merge_results(vector_results, keyword_results)

    if not merged_docs and not wiki_docs:
        print("No relevant context found for your question.")
        return ""

    context_chunks = []
    metadatas = []

    if merged_docs:
        #RERANK
        raw_context_chunks, raw_metadatas = rerank(question, merged_docs, merged_meta)

        # Prefer wiki context first; fill remaining slots with raw context.
        context_chunks = wiki_docs + raw_context_chunks[:max(0, 4 - len(wiki_docs))]
        metadatas = wiki_meta + raw_metadatas[:max(0, 4 - len(wiki_meta))]
    else:
        context_chunks = wiki_docs
        metadatas = wiki_meta

    # -------------------------
    # DEBUG OUTPUT
    # -------------------------
    print("\n--- Retrieved Context (Wiki + Hybrid Search) ---\n")

    sources = []

    for i, (chunk, meta) in enumerate(zip(context_chunks, metadatas)):

        source = meta.get("source", "unknown") if meta else "unknown"
        doc_type = meta.get("type", "unknown") if meta else "unknown"

        sources.append(source)

        print(f"Chunk {i+1}")
        print(f"Source: {source}")
        print(f"Type: {doc_type}")
        # print("\nText:")
        # print(chunk[:100])
        # print("\n--------------------------\n")

    # -------------------------
    # PROMPT
    # -------------------------

    # Label each chunk with its source so the model can cite precisely.
    numbered_context = "\n\n".join(
        f"[{i+1}] ({meta.get('source', 'unknown') if meta else 'unknown'})\n{chunk}"
        for i, (chunk, meta) in enumerate(zip(context_chunks, metadatas))
    )

    # Deduplicated, human-readable source list.
    unique_sources = list(dict.fromkeys(sources))
    sources_formatted = "\n".join(f"- {s}" for s in unique_sources)

    prompt = f"""You are a precise assistant for a personal knowledge base built with the Zettelkasten method.

INSTRUCTIONS:
- Answer using ONLY the numbered context passages below.
- If the context is insufficient, say "I don't have enough information in my notes to answer this."
- Be concise and direct. Avoid unnecessary filler.
- Always cite the source file and passage number when you use context, e.g. [1 - source.md], [2 - other-file.md].

CONTEXT:
{numbered_context}

QUESTION:
{question}

SOURCES AVAILABLE:
{sources_formatted}

ANSWER:"""
    
    # -------------------------
    # OLLAMA CALL
    # -------------------------
    print("\nAI: ", end="", flush=True)
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": config.LLM_MODEL,
            "prompt": prompt,
            "stream": True
        },
        stream=True
    )
    full_response = ""
    
    for line in response.iter_lines():

        if line:
            data = json.loads(line)
            token = data.get("response", "")

            print(token, end="", flush=True)

            full_response += token

    print("\n")
    return full_response

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

        #print("\nAnswer:\n")
        #print(answer)


if __name__ == "__main__":
    chat()