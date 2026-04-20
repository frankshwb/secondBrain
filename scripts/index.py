import os
import sys
import re
import json
import tempfile
from datetime import datetime

# ermöglicht Import von config.py aus Root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import requests
from tqdm import tqdm
import hashlib


# -------------------------
# HASH FUNCTION for Duplicate-Safe Indexing with Idempotency Layer
# -------------------------
def make_id(text, source):
    return hashlib.md5((text + source).encode()).hexdigest()


def slugify_filename(name):
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower()
    return slug or "untitled"


def wiki_path_guard(path):
    wiki_root = os.path.abspath(config.WIKI_PATH)
    candidate = os.path.abspath(path)
    return candidate == wiki_root or candidate.startswith(wiki_root + os.sep)


def safe_atomic_write(path, content):
    if not wiki_path_guard(path):
        raise ValueError(f"Refusing to write outside wiki root: {path}")

    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)

    temp_file = None
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, dir=directory, encoding="utf-8") as tmp:
            tmp.write(content)
            temp_file = tmp.name
        os.replace(temp_file, path)
    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)


def read_text_if_exists(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def ensure_wiki_structure():
    os.makedirs(config.WIKI_PATH, exist_ok=True)
    os.makedirs(config.WIKI_PAGES_PATH, exist_ok=True)
    os.makedirs(config.WIKI_SOURCES_PATH, exist_ok=True)
    os.makedirs(config.WIKI_SYSTEM_PATH, exist_ok=True)

    if not os.path.exists(config.WIKI_SCHEMA_FILE):
        safe_atomic_write(
            config.WIKI_SCHEMA_FILE,
            "# Wiki Schema\n\n"
            "- Generated wiki content lives in this folder only.\n"
            "- `sources/` stores source-oriented summaries.\n"
            "- `pages/` stores topic/entity pages referenced by wikilinks.\n"
            "- `system/index.md` is the content catalog.\n"
            "- `system/log.md` is append-only operation log.\n",
        )

    if not os.path.exists(config.WIKI_INDEX_FILE):
        safe_atomic_write(config.WIKI_INDEX_FILE, "# Wiki Index\n\n")

    if not os.path.exists(config.WIKI_LOG_FILE):
        safe_atomic_write(config.WIKI_LOG_FILE, "# Wiki Log\n\n")


def call_ollama_wiki_summarizer(doc_name, doc_type, text_excerpt):
    prompt = f"""You are maintaining an Obsidian LLM wiki.

Write markdown only.
Keep output concise and grounded in the source text.
Use Obsidian wikilinks like [[Concept Name]] for relevant entities/topics.

Source file: {doc_name}
Source type: {doc_type}

Produce these sections exactly:
# Summary
## Key Points
## Entities and Concepts
## Open Questions

Source text excerpt:
{text_excerpt}
"""

    try:
        response = requests.post(
            f"{config.OLLAMA_URL}/api/generate",
            json={
                "model": config.LLM_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=90,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except Exception as e:
        print(f"Warning: Wiki summary generation failed for '{doc_name}': {e}")
        return ""


def extract_wikilinks(markdown_text):
    links = re.findall(r"\[\[([^\]|#]+)(?:\|[^\]]+)?\]\]", markdown_text)
    cleaned = []
    for link in links:
        value = link.strip()
        if value:
            cleaned.append(value)
    return sorted(set(cleaned))


def write_source_wiki_page(doc_name, doc_path, doc_type, generated_markdown, fallback_excerpt):
    title = os.path.splitext(os.path.basename(doc_name))[0]
    filename = f"{slugify_filename(title)}.md"
    source_page_path = os.path.join(config.WIKI_SOURCES_PATH, filename)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    body = generated_markdown.strip() if generated_markdown.strip() else (
        "# Summary\n"
        "No model-generated summary available.\n\n"
        "## Key Points\n"
        f"- Source ingested: `{doc_name}`\n\n"
        "## Entities and Concepts\n"
        "- None extracted.\n\n"
        "## Open Questions\n"
        "- What additional context should be added for this source?\n"
    )

    page_content = (
        "---\n"
        f"source_file: {doc_name}\n"
        f"source_path: {doc_path}\n"
        f"source_type: {doc_type}\n"
        f"updated_at: {now}\n"
        "---\n\n"
        f"# Source: {doc_name}\n\n"
        f"Original path: `{doc_path}`\n\n"
        f"{body}\n\n"
        "## Raw Excerpt\n"
        f"{fallback_excerpt}\n"
    )

    safe_atomic_write(source_page_path, page_content)
    return source_page_path


def upsert_topic_pages(topic_names, source_page_path):
    touched = []
    source_stem = os.path.splitext(os.path.basename(source_page_path))[0]
    source_link = f"[[sources/{source_stem}|{source_stem}]]"

    for topic in topic_names:
        topic_file = os.path.join(config.WIKI_PAGES_PATH, f"{slugify_filename(topic)}.md")
        existing = read_text_if_exists(topic_file)

        if not existing:
            content = (
                f"# {topic}\n\n"
                "## Mentions\n"
                f"- Referenced from {source_link}\n"
            )
            safe_atomic_write(topic_file, content)
            touched.append(topic_file)
            continue

        mention_line = f"- Referenced from {source_link}"
        if mention_line not in existing:
            updated = existing.rstrip() + "\n" + mention_line + "\n"
            safe_atomic_write(topic_file, updated)
            touched.append(topic_file)

    return touched


def rebuild_wiki_index():
    lines = ["# Wiki Index", ""]

    def collect_entries(base_dir, section_title):
        lines.append(f"## {section_title}")
        files = sorted([f for f in os.listdir(base_dir) if f.endswith(".md")])
        if not files:
            lines.append("- (none)")
            lines.append("")
            return

        for file_name in files:
            absolute = os.path.join(base_dir, file_name)
            rel = os.path.relpath(absolute, config.WIKI_PATH).replace("\\", "/")
            content = read_text_if_exists(absolute)
            first_heading = ""
            for line in content.splitlines():
                if line.startswith("# "):
                    first_heading = line[2:].strip()
                    break
            summary = first_heading if first_heading else file_name
            lines.append(f"- [[{rel}|{summary}]]")
        lines.append("")

    collect_entries(config.WIKI_SOURCES_PATH, "Sources")
    collect_entries(config.WIKI_PAGES_PATH, "Pages")

    safe_atomic_write(config.WIKI_INDEX_FILE, "\n".join(lines).rstrip() + "\n")


def append_wiki_log(action, doc_name, touched_files):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    touched_rel = [os.path.relpath(path, config.WIKI_PATH).replace("\\", "/") for path in touched_files]

    entry_lines = [
        f"## [{timestamp}] {action} | {doc_name}",
        "",
        "Touched files:",
    ]
    for rel in touched_rel:
        entry_lines.append(f"- `{rel}`")
    entry_lines.append("")

    existing = read_text_if_exists(config.WIKI_LOG_FILE)
    updated = existing.rstrip() + "\n\n" + "\n".join(entry_lines) + "\n"
    safe_atomic_write(config.WIKI_LOG_FILE, updated)


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

            try:
                reader = PdfReader(path)
                text = ""
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted
            except Exception as e:
                print(f"Warning: Could not read PDF '{file}': {e}")
                continue

            if text.strip():
                doc_path = os.path.relpath(path, config.VAULT_PATH).replace("\\", "/")
                docs.append({
                    "name": file,
                    "text": text,
                    "path": doc_path,
                    "doc_type": "pdf",
                })

    return docs


# -------------------------
# LOAD MARKDOWN NOTES
# -------------------------
def load_markdown():

    docs = []

    wiki_root = os.path.abspath(config.WIKI_PATH)

    for root, dirs, files in os.walk(config.VAULT_PATH):

        # Skip generated llm-wiki folder so generated pages are not re-ingested
        # as raw source files.
        dirs[:] = [
            d for d in dirs
            if not os.path.abspath(os.path.join(root, d)).startswith(wiki_root + os.sep)
            and os.path.abspath(os.path.join(root, d)) != wiki_root
        ]

        for file in files:

            if file.endswith(".md"):

                path = os.path.join(root, file)
                print("md file: ", path)

                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                    if text.strip():
                        doc_path = os.path.relpath(path, config.VAULT_PATH).replace("\\", "/")
                        docs.append({
                            "name": file,
                            "text": text,
                            "path": doc_path,
                            "doc_type": "markdown",
                        })
                except Exception as e:
                    print(f"Warning: Could not read markdown '{file}': {e}")
                    continue

    return docs


# -------------------------
# CHUNKING
# -------------------------
def chunk_text(text, max_chars=1200, overlap=200):

    # Split on paragraph boundaries first, then sentence-split within each paragraph.
    # This preserves list items, code blocks, and structured notes better than a
    # flat sentence split on the entire document.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    sentences = []
    for paragraph in paragraphs:
        parts = paragraph.replace("\n", " ").split(". ")
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        if len(current_chunk) + len(sentence) > max_chars:

            chunks.append(current_chunk.strip())

            # overlap behalten (wichtiger Kontext)
            current_chunk = current_chunk[-overlap:] + " " + sentence

        else:
            current_chunk += " " + sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# -------------------------
# INDEXING PIPELINE
# -------------------------
def index_documents():

    ensure_wiki_structure()

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

    for doc in tqdm(docs):

        doc_name = doc["name"]
        text = doc["text"]
        doc_path = doc["path"]
        doc_type = doc["doc_type"]

        chunks = chunk_text(text)

        if not chunks:
            continue

        embeddings = embed_model.encode(chunks)

        for i, chunk in enumerate(chunks):

            # upsert makes the overwrite-on-same-id behaviour explicit:
            # re-indexing the same content updates the stored chunk in place
            # rather than raising a DuplicateIDError.
            collection.upsert(
                documents=[chunk],
                embeddings=[embeddings[i].tolist()],
                ids=[make_id(chunk, doc_name)],
                metadatas=[{
                    "source": doc_name,
                    "chunk_id": i,
                    "type": doc_type,
                    "path": doc_path
                }]
            )

        excerpt = "\n\n".join(chunks[:4])[:6000]
        generated_markdown = call_ollama_wiki_summarizer(doc_name, doc_type, excerpt)
        source_page = write_source_wiki_page(
            doc_name=doc_name,
            doc_path=doc_path,
            doc_type=doc_type,
            generated_markdown=generated_markdown,
            fallback_excerpt=excerpt,
        )

        topic_names = extract_wikilinks(generated_markdown)
        topic_pages = upsert_topic_pages(topic_names, source_page)

        touched = [source_page] + topic_pages
        append_wiki_log("ingest", doc_name, touched)

    rebuild_wiki_index()

    print("Indexing finished.")


if __name__ == "__main__":
    index_documents()