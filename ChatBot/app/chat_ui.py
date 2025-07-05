import streamlit as st
from conversational_bot_with_memory import ChatApp
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="User Chatbot", layout="wide")

st.title("📊 Chat Assistant with Memory")

# Sidebar user setup
user_id = st.sidebar.text_input("User ID", value="user001")
model = st.sidebar.selectbox("Model", options=["gpt-4o-mini", "gpt-4o"], index=0)
reset = st.sidebar.button("Reset Conversation")

# Store conversation in session state
if "history" not in st.session_state:
    st.session_state["history"] = []

# Reset session and DB memory
if reset:
    from conversational_bot_with_memory import SessionLocal, MessageStore
    with SessionLocal() as session:
        session.query(MessageStore).filter_by(user_id=user_id).delete()
        session.commit()
    st.session_state["history"] = []
    st.success("Conversation reset.")

# Instruction setup
with st.expander("System Instruction (Optional)", expanded=True):
    system_prompt = st.text_area(
        "System message to guide behavior",
        value="You are a helpful data analyst who generates SQL queries for users...",
        height=100
    )
    if st.button("Set Instruction"):
        app = ChatApp(model_name=model, user_id=user_id, user_role="system", user_text=system_prompt)
        app.run()
        st.success("Instruction sent!")

# User chat input
query = st.chat_input("Ask a data question...")
if query:
    app = ChatApp(model_name=model, user_id=user_id, user_role="user", user_text=query)
    result = app.run()

    # Store and show
    st.session_state["history"].append((query, result["output"]))

# Display history
for user_msg, bot_msg in st.session_state["history"]:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg)
