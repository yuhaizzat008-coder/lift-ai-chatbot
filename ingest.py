import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DOCS_FOLDER = "docs"
DB_FOLDER = "chroma_db"

def ingest_documents():
    pdf_files = [f for f in os.listdir(DOCS_FOLDER) if f.endswith(".pdf")]
    all_docs = []

    for pdf in pdf_files:
        loader = PyPDFLoader(os.path.join(DOCS_FOLDER, pdf))
        docs = loader.load()
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    # FREE Embeddings (no OpenAI needed)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=DB_FOLDER)
    vectordb.persist()

    print("Ingestion complete with FREE embeddings!")

if __name__ == "__main__":
    ingest_documents()
