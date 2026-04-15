import os
import sys

# ermöglicht Import von config.py aus Root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm
import hashlib


# -------------------------
# HASH FUNCTION for Duplicate-Safe Indexing with Idempotency Layer
# -------------------------
def make_id(text, source):
    return hashlib.md5((text + source).encode()).hexdigest()


# -------------------------
# LOAD PDF DOCUMENTS
# -------------------------
def load_pdfs():

    docs = []

    if not os.path.exists(config.PDF_PATH):
        print("PDF folder not found:", config.PDF_PATH)
        return docs

    for file in os.listdir(config.PDF_PATH):

        if file.endswith(".pdf"):

            path = os.path.join(config.PDF_PATH, file)
            print("pdf file: ", path)

            reader = PdfReader(path)

            text = ""

            for page in reader.pages:

                extracted = page.extract_text()

                if extracted:
                    text += extracted

            if text.strip():
                docs.append((file, text))

    return docs


# -------------------------
# LOAD MARKDOWN NOTES
# -------------------------
def load_markdown():

    docs = []

    for root, dirs, files in os.walk(config.VAULT_PATH):

        for file in files:

            if file.endswith(".md"):

                path = os.path.join(root, file)
                print("md file: ", path)

                with open(path, "r", encoding="utf-8") as f:

                    text = f.read()

                    if text.strip():

                        docs.append((file, text))

    return docs


# -------------------------
# CHUNKING
# -------------------------
def chunk_text(text, max_chars=1200, overlap=200):

    sentences = text.replace("\n", " ").split(". ")

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        # wenn Chunk zu groß wird → speichern
        if len(current_chunk) + len(sentence) > max_chars:

            chunks.append(current_chunk.strip())

            # overlap behalten (wichtiger Kontext)
            current_chunk = current_chunk[-overlap:] + " " + sentence

        else:
            current_chunk += " " + sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# -------------------------
# INDEXING PIPELINE
# -------------------------
def index_documents():

    print("Loading embedding model...")

    embed_model = SentenceTransformer(config.EMBEDDING_MODEL)

    print("Connecting to Chroma...")

    client = chromadb.PersistentClient(path=config.CHROMA_PATH)

    collection = client.get_or_create_collection("second_brain")

    print("Loading PDFs...")

    pdf_docs = load_pdfs()

    print("Loading Markdown notes...")

    md_docs = load_markdown()

    docs = pdf_docs + md_docs

    print(f"Total documents found: {len(docs)}")

    for doc_name, text in tqdm(docs):

        chunks = chunk_text(text)

        embeddings = embed_model.encode(chunks)

        for i, chunk in enumerate(chunks):
            
            collection.add(
                documents=[chunk],
                embeddings=[embeddings[i].tolist()],
                ids=[make_id(chunk, doc_name)],
                metadatas=[{
                    "source": doc_name,
                    "chunk_id": i,
                    "type": "pdf" if doc_name.endswith(".pdf") else "markdown",
                    "path": doc_name
                }]
)

    print("Indexing finished.")


if __name__ == "__main__":
    index_documents()