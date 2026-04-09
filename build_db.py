import os
import shutil

from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# ---------------------------
# Paths
# ---------------------------
PROJECT_DIR = r"C:\Users\User\Desktop\lift_ai_project"
DOCS_DIR = os.path.join(PROJECT_DIR, "docs")
DB_DIR = os.path.join(PROJECT_DIR, "db")

# ---------------------------
# 1) Load documents (PDF + TXT/MD)
# ---------------------------
def load_docs(docs_dir: str):
    docs = []

    # Load PDFs
    pdf_loader = DirectoryLoader(
        docs_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    docs.extend(pdf_loader.load())

    # Load TXT/MD (optional)
    text_loader = DirectoryLoader(
        docs_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
        use_multithreading=True,
    )
    docs.extend(text_loader.load())

    md_loader = DirectoryLoader(
        docs_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
        use_multithreading=True,
    )
    docs.extend(md_loader.load())

    return docs

# ---------------------------
# 2) Split into chunks
# ---------------------------
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", "•", "-", ".", " ", ""],
    )
    return splitter.split_documents(docs)

# ---------------------------
# 3) Build Chroma DB
# ---------------------------
def build_db():
    if not os.path.exists(DOCS_DIR):
        raise FileNotFoundError(f"Docs folder not found: {DOCS_DIR}")

    # Reset DB to avoid embedding mismatch / stale data
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    print(f"Loading docs from: {DOCS_DIR}")
    docs = load_docs(DOCS_DIR)
    print(f"Loaded documents: {len(docs)}")

    if len(docs) == 0:
        print("No documents found. Make sure your PDFs/TXT/MD are inside the docs folder.")
        return

    print("Splitting into chunks...")
    chunks = split_docs(docs)
    print(f"Total chunks: {len(chunks)}")

    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print(f"Building Chroma DB at: {DB_DIR}")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    db.persist()

    print("✅ Done! Chroma DB created.")
    print("Tip: If you add new documents later, run build_db.py again.")

if __name__ == "__main__":
    build_db()
