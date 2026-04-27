import streamlit as st
import os

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Lift Safety Assistant", layout="wide")

st.title("🛗 Lift Safety Assistant")
st.markdown("AI-powered safety guidance for lift maintenance and inspection")

# -------------------------------
# SAFETY DISCLAIMER
# -------------------------------
st.warning("⚠️ This system provides guidance only. Always follow certified safety procedures.")

# -------------------------------
# OPENAI API KEY
# -------------------------------
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("OpenAI API key not found. Add it in Streamlit secrets.")
    st.stop()

# -------------------------------
# LOAD EMBEDDINGS
# -------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -------------------------------
# LOAD VECTOR DATABASE
# -------------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = load_embeddings()
    try:
        db = FAISS.load_local("vectorstore", embeddings)
        return db
    except:
        st.error("Vector database not found. Ensure 'vectorstore' folder exists.")
        return None

# -------------------------------
# LOAD OPENAI MODEL
# -------------------------------
@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2
    )

# -------------------------------
# BUILD RAG SYSTEM
# -------------------------------
@st.cache_resource
def build_qa():
    db = load_vectorstore()
    if db is None:
        return None

    llm = load_llm()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )
    return qa

qa_system = build_qa()

# -------------------------------
# SAFETY RESPONSE FORMAT
# -------------------------------
def format_response(answer):
    checklist = (
        "SAFETY CHECKLIST:\n"
        "- Isolate power supply\n"
        "- Apply lockout/tagout (LOTO)\n"
        "- Wear PPE (helmet, gloves, harness)\n"
        "- Secure working area\n"
    )

    warning = (
        "⚠️ WARNING: Ensure lift is fully isolated before inspection.\n"
        "⚠️ WARNING: Ensure lift is fully isolated before inspection.\n"
    )

    return f"{checklist}\n\n{warning}\n\nTECHNICAL GUIDANCE:\n{answer}"

# -------------------------------
# USER INPUT
# -------------------------------
query = st.text_input("Enter your lift safety question:")

if query:
    if qa_system is None:
        st.error("System not ready. Missing vector database.")
    else:
        with st.spinner("Processing safety query..."):
            result = qa_system(query)

            answer = result["result"]
            sources = result["source_documents"]

            formatted = format_response(answer)

            st.markdown("### 🧠 Response")
            st.text(formatted)

            # -------------------------------
            # SHOW SOURCES (TRACEABILITY)
            # -------------------------------
            with st.expander("📄 Retrieved Safety References"):
                for i, doc in enumerate(sources):
                    st.write(f"Source {i+1}:")
                    st.write(doc.page_content[:300])

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Lift Safety AI Assistant | RAG + OpenAI Powered")
