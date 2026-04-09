import os
import time
import json
import streamlit as st

from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from langchain_classic.chains import RetrievalQA
from langchain_classic.memory import ConversationBufferMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate


# =========================
# CONFIG
# =========================
PROJECT_DIR = r"C:\Users\User\Desktop\lift_ai_project"
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "tinyllama-1b.gguf")
DB_FOLDER = os.path.join(PROJECT_DIR, "chroma_db")
USER_FILE = "users.json"

os.makedirs(DB_FOLDER, exist_ok=True)


# =========================
# USER SYSTEM
# =========================
if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump({}, f)


def load_users():
    with open(USER_FILE, "r") as f:
        return json.load(f)


def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


def login_signup():
    option = st.selectbox("Select Option", ["Login", "Sign Up"])
    users = load_users()

    if option == "Login":
        st.title("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username in users and users[username] == password:
                st.session_state.logged_in = True
                st.session_state.user = username
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid username or password")

    else:
        st.title("📝 Sign Up")
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")

        if st.button("Create Account"):
            if new_user in users:
                st.warning("Username already exists")
            elif new_user == "" or new_pass == "":
                st.warning("Fill all fields")
            else:
                users[new_user] = new_pass
                save_users(users)
                st.success("Account created! Please login")


if not st.session_state.logged_in:
    login_signup()
    st.stop()


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_llm():
    return LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=2048,
        max_tokens=512,
        temperature=0.2,
        n_threads=6,
        n_gpu_layers=0,
        verbose=False
    )

llm = load_llm()


# =========================
# LOAD VECTOR DB
# =========================
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=DB_FOLDER,
        embedding_function=embeddings
    )

    return db, embeddings


vectordb, embeddings = load_db()
retriever = vectordb.as_retriever(search_kwargs={"k": 3})


# =========================
# MEMORY
# =========================
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="result"
)


# =========================
# PROMPT (IMPORTANT)
# =========================
PROMPT_TEMPLATE = """
You are a lift maintenance and inspection expert.

RULES:
- Answer clearly and professionally
- DO NOT copy raw text
- Summarize information
- Overspeed governor = speed only (NOT weight)

FORMAT:

Answer:
- explanation

Inspection Guidance:
- steps

Safety Note:
- importance

Context:
{context}

Question:
{question}
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
    memory=memory
)


# =========================
# WEB SEARCH
# =========================
search = DuckDuckGoSearchAPIWrapper()


def smart_answer(query):

    try:
        result = qa_chain({"query": query})
        local_answer = result["result"]

        if "cannot confirm" in local_answer.lower() or len(local_answer) < 80:
            raise Exception("Weak answer")

        return local_answer

    except:
        web_result = search.run(query)

        prompt = f"""
Answer clearly using this information.

DO NOT copy text directly.

{web_result}

Question: {query}

FORMAT:

Answer:
- explanation

Inspection Guidance:
- steps

Safety Note:
- importance
"""

        response = llm.invoke(prompt)

        return response if isinstance(response, str) else response.content


# =========================
# CLEAN ANSWER
# =========================
def clean_answer(answer):

    if "shall" in answer.lower():
        return """Answer:
- If a lift overspeeds, the overspeed governor detects the speed and activates the safety gear to stop the elevator.

Inspection Guidance:
- Test governor regularly
- Check safety gear

Safety Note:
- Critical for safety
"""

    if len(answer) > 700:
        return answer[:500] + "..."

    return answer


# =========================
# UI
# =========================
st.title("🛗 Lift Inspection AI System")
st.sidebar.write("Logged in as:", st.session_state.user)


# =========================
# PDF UPLOAD
# =========================
st.sidebar.header("Upload PDF")

pdf_files = st.sidebar.file_uploader(
    "Upload PDF", type=["pdf"], accept_multiple_files=True
)

if pdf_files:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for file in pdf_files:
        path = os.path.join(PROJECT_DIR, file.name)

        with open(path, "wb") as f:
            f.write(file.getbuffer())

        loader = PyPDFLoader(path)
        docs = loader.load()

        chunks = splitter.split_documents(docs)
        vectordb.add_documents(chunks)

    vectordb.persist()
    st.sidebar.success("PDF added!")


# =========================
# IMAGE UPLOAD
# =========================
image = st.sidebar.file_uploader("Upload Image", type=["png", "jpg"])

if image:
    st.sidebar.image(image, caption="Uploaded Image")


# =========================
# CHAT UI
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


query = st.chat_input("Ask about lift maintenance or inspection...")

if query:

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    answer = smart_answer(query)
    answer = clean_answer(answer)

    with st.chat_message("assistant"):

        placeholder = st.empty()
        text = ""

        for word in answer.split():
            text += word + " "
            placeholder.markdown(text + "▌")
            time.sleep(0.02)

        placeholder.markdown(text)

    st.session_state.messages.append({"role": "assistant", "content": answer})