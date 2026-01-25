# streamlit_app.py
import streamlit as st
from src.chains import medical_chain

st.set_page_config(page_title="Medical Chatbot", page_icon="🩺")
st.title("🩺 Medical Chatbot")

# Simple session
if "session_id" not in st.session_state:
    st.session_state.session_id = "user1"

# Chat history display
if "history" not in st.session_state:
    st.session_state.history = []

# User input
user_question = st.text_input("Ask a medical question:")

if st.button("Send") and user_question:
    # Add user question to history
    st.session_state.history.append({"type": "human", "content": user_question})

    # Run medical_chain
    response = medical_chain.invoke({
        "question": user_question,
        "history": st.session_state.history,
        "session_id": st.session_state.session_id
    })

    # Add bot response to history
    st.session_state.history.append({"type": "ai", "content": response})

# Display conversation
st.subheader("Conversation:")
for message in st.session_state.history:
    if message["type"] == "human":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Bot:** {message['content']}")
