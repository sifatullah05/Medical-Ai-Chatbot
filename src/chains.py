from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda
from src.vectorstore import retriever
import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in .env!")

chat_model = ChatGroq(
    model = "openai/gpt-oss-20b",
    temperature = 0
)

SYSTEM_PROMPT = """
You are a medical information assistant.

Your role is to provide accurate, educational medical information
based ONLY on the provided context.

CONVERSATION HISTORY:
{history}

MEDICAL CONTEXT:
{context}

STRICT RULES:
- Do NOT provide medical diagnosis.
- Do NOT recommend treatments, procedures, or medication dosages.
- Do NOT make clinical decisions.
- If the information is not in the context, say "I don't know based on the provided information."
- Always encourage consulting a licensed medical professional.

STYLE RULES:
- Be concise and factual.
- Maximum 3 sentences.
- No speculation.
- No alarmist language.

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

store = {}

def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
def build_context(question: str):
    docs = retriever.invoke(question)
    return "\n".join(doc.page_content for doc in docs)
def format_history(messages):
    if not messages:
        return "No previous conversation."
    return "\n".join(f"{m.type.capitalize()}: {m.content}" for m in messages)


base_chain = (
    {
        "question": RunnableLambda(lambda x: x["question"]),
        "context": RunnableLambda(lambda x: build_context(x["question"])),
        "history": RunnableLambda(
            lambda x: format_history(x["history"])  
        ),
    }
    | prompt
    | chat_model
    | StrOutputParser()
)

medical_chain = RunnableWithMessageHistory(
    base_chain,
    get_history,
    input_messages_key="question",
    history_messages_key="history",
)
