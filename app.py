import os
import json
import time
import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================
# CONFIG
# =========================
DB_FOLDER = "chroma_db"
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
# VECTOR DATABASE
# =========================
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        persist_directory=DB_FOLDER,
        embedding_function=embeddings
    )


vectordb = load_db()


# =========================
# WEB SEARCH
# =========================
search = DuckDuckGoSearchAPIWrapper()


# =========================
# AI ENGINE
# =========================
def ai_answer(query):

    q = query.lower()

    # -------- RULE-BASED (ACCURATE CORE KNOWLEDGE) --------
    if "overspeed governor" in q:
        return """Answer:
- Overspeed governor monitors elevator speed and activates safety gear when speed exceeds safe limits.

Inspection Guidance:
- Test governor regularly
- Check safety gear operation

Safety Note:
- Prevents dangerous overspeed accidents
"""

    if "overspeed" in q:
        return """Answer:
- If a lift overspeeds, the overspeed governor detects it and stops the lift using safety gear.

Inspection Guidance:
- Inspect governor and braking system
- Ensure proper calibration

Safety Note:
- Prevents free fall accidents
"""

    # -------- LOCAL PDF SEARCH --------
    docs = vectordb.similarity_search(query, k=2)

    if docs and len(docs[0].page_content) > 50:
        return f"""Answer:
- {docs[0].page_content[:200]}

Inspection Guidance:
- Refer to maintenance procedures

Safety Note:
- Follow safety standards
"""

    # -------- WEB SEARCH FALLBACK --------
    try:
        web = search.run(query)

        return f"""Answer:
- {web[:300]}

Inspection Guidance:
- Verify information with official standards

Safety Note:
- Always follow safety regulations
"""

    except:
        return """Answer:
- Unable to retrieve information.

Inspection Guidance:
- Check manuals or standards

Safety Note:
- Follow safety procedures
"""


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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    for file in pdf_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())

        loader = PyPDFLoader(file.name)
        docs = loader.load()

        chunks = splitter.split_documents(docs)
        vectordb.add_documents(chunks)

    vectordb.persist()
    st.sidebar.success("PDF added successfully!")


# =========================
# CHAT MEMORY
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =========================
# CHAT INPUT
# =========================
query = st.chat_input("Ask about lift maintenance or inspection...")

if query:

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    answer = ai_answer(query)

    with st.chat_message("assistant"):

        placeholder = st.empty()
        text = ""

        for word in answer.split():
            text += word + " "
            placeholder.markdown(text + "▌")
            time.sleep(0.02)

        placeholder.markdown(text)

    st.session_state.messages.append({"role": "assistant", "content": answer})
