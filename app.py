       import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Lift Safety Assistant", page_icon="🛗", layout="wide")

# =========================
# SESSION STATE (CHAT MEMORY)
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================
# SIDEBAR (RESET BUTTON)
# =========================
with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("🔄 Reset Chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.subheader("📌 Example Questions")
    st.write("• What is lockout-tagout?")
    st.write("• Lift emergency procedure")
    st.write("• Electrical safety in lifts")

# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """
You are a certified lift safety inspector.

Respond using STRICT structure:
1. Safety Checklist
2. Critical Warning (repeat twice)
3. Technical Steps
4. Diagnostic Questions

Rules:
- Keep sentences SHORT
- Be DIRECT and COMMANDING
- No polite phrases
- No extra explanation
"""

# =========================
# LOAD MODEL (CACHED)
# =========================
@st.cache_resource
def load_llm():
    return CTransformers(
        model="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        model_type="llama",
        config={
            'max_new_tokens': 180,
            'temperature': 0.2
        }
    )

# =========================
# LOAD VECTOR DATABASE (CACHED)
# =========================
@st.cache_resource
def load_vectorstore():
    loader = TextLoader("data.txt")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)

    return db

# =========================
# INITIALIZE SYSTEM
# =========================
llm = load_llm()
vectorstore = load_vectorstore()

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# =========================
# HEADER
# =========================
st.title("🛗 Lift Safety AI Assistant")
st.caption("Real-time safety guidance based on ISO / EN / ASME standards")

# =========================
# DISPLAY CHAT HISTORY
# =========================
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])

# =========================
# USER INPUT
# =========================
query = st.chat_input("Ask your safety question...")

if query:
    # Show user message
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing safety procedure..."):
            result = qa({"query": SYSTEM_PROMPT + "\n" + query})

            response = result["result"]
            st.write(response)

            # SOURCE DISPLAY
            with st.expander("📄 Source Reference"):
                for doc in result["source_documents"]:
                    st.write(doc.page_content)

    # Save assistant response
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("⚠️ Safety guidance only. Verify with official standards before real-world use.")
