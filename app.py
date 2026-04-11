 import json
import time
import streamlit as st
from duckduckgo_search import DDGS


# =========================
# USER SYSTEM
# =========================
USER_FILE = "users.json"

try:
    with open(USER_FILE, "r") as f:
        users = json.load(f)
except:
    users = {}

def save_users():
    with open(USER_FILE, "w") as f:
        json.dump(users, f)


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


def login_signup():
    option = st.selectbox("Select Option", ["Login", "Sign Up"])

    if option == "Login":
        st.title("🔐 Login")
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if u in users and users[u] == p:
                st.session_state.logged_in = True
                st.session_state.user = u
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid login")

    else:
        st.title("📝 Sign Up")
        u = st.text_input("New Username")
        p = st.text_input("New Password", type="password")

        if st.button("Create Account"):
            if u in users:
                st.warning("User exists")
            elif u == "" or p == "":
                st.warning("Fill all fields")
            else:
                users[u] = p
                save_users()
                st.success("Account created")


if not st.session_state.logged_in:
    login_signup()
    st.stop()


# =========================
# SIMPLE AI ENGINE
# =========================
def web_search(query):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
        return " ".join([r["body"] for r in results])


def ai_answer(query):

    q = query.lower()

    # ---- RULE BASED ----
    if "overspeed governor" in q:
        return """Answer:
- Overspeed governor detects excessive speed and activates safety gear.

Inspection Guidance:
- Test governor regularly
- Check safety gear

Safety Note:
- Prevents dangerous accidents
"""

    if "overspeed" in q:
        return """Answer:
- If a lift overspeeds, the governor stops the lift automatically.

Inspection Guidance:
- Inspect braking system
- Check calibration

Safety Note:
- Prevents free fall
"""

    # ---- WEB SEARCH ----
    try:
        web = web_search(query)

        return f"""Answer:
- {web[:300]}

Inspection Guidance:
- Verify from official sources

Safety Note:
- Follow safety standards
"""
    except:
        return "System unable to fetch data."


# =========================
# UI
# =========================
st.title("🛗 Lift Inspection AI System")
st.sidebar.write("User:", st.session_state.user)


# =========================
# CHAT
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


query = st.chat_input("Ask about lift maintenance...")

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
