
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# -----------------------------
# LLM (streaming enabled)
# -----------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    streaming=True
)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("HM-GPT")
st.markdown("Your questions will be answered **with full conversation history** and **streaming response**.")

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Optional: System prompt (always at start)
SYSTEM_PROMPT = SystemMessage(content="You are a helpful AI assistant. Answer all questions based on the conversation history.")

# Display chat history
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).markdown(msg.content)

# Input
query = st.chat_input("Ask anything:")

if query:
    # Add user message to history
    user_msg = HumanMessage(content=query)
    st.session_state.messages.append(user_msg)
    st.chat_message("user").markdown(query)

    # Prepare messages to send: system prompt + history
    messages_to_send = [SYSTEM_PROMPT] + st.session_state.messages

    # Assistant streaming response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        # Stream tokens from LLM
        for chunk in llm.stream(messages_to_send):
            if chunk.content:
                full_response += chunk.content
                placeholder.markdown(full_response)

    # Save assistant response in history
    st.session_state.messages.append(AIMessage(content=full_response))

