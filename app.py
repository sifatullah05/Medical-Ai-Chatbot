import streamlit as st
from datetime import datetime
from dotenv import load_dotenv


# --- Load environment variables ---
load_dotenv()

# --- Import medical_chain ---
from src.chains import medical_chain  # src ফোল্ডারের chains.py থেকে import

# --- Streamlit page config ---
st.set_page_config(page_title="Medical Chatbot", page_icon="💬", layout="centered")

# --- Session state to store messages ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Function to add a user message and bot response ---
def add_message(user_text):
    st.session_state.messages.append({"sender": "user", "text": user_text})
    
    # Real bot response using medical_chain
    response = medical_chain.invoke({"question": user_text})
    st.session_state.messages.append({"sender": "bot", "text": response})

# --- Sidebar / Clear chat ---
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []

# --- Chat UI ---
st.markdown("<h2 style='text-align:center; color:white;'>Medical Chatbot</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:rgba(255,255,255,0.6);'>Ask me anything!</p>", unsafe_allow_html=True)

chat_container = st.container()

# --- Display messages ---
with chat_container:
    for msg in st.session_state.messages:
        time_str = datetime.now().strftime("%H:%M")
        if msg["sender"] == "user":
            st.markdown(f"""
            <div style='text-align:right; margin:5px 0;'>
                <div style='display:inline-block; background-color:#58cc71; color:white; padding:10px 15px; border-radius:20px; max-width:70%; word-wrap:break-word;'>
                    {msg['text']} <span style='font-size:10px; color:rgba(255,255,255,0.6);'>{time_str}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='text-align:left; margin:5px 0;'>
                <div style='display:inline-block; background-color:#52acff; color:white; padding:10px 15px; border-radius:20px; max-width:70%; word-wrap:break-word;'>
                    {msg['text']} <span style='font-size:10px; color:rgba(255,255,255,0.6);'>{time_str}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- Input box ---
with st.form(key="message_form", clear_on_submit=True):
    user_input = st.text_input("Type your message...", "")
    submit_button = st.form_submit_button("Send")

    if submit_button and user_input:
        add_message(user_input)
        st.experimental_rerun()

