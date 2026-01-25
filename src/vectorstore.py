import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from src.loader import *


load_dotenv()

def huggingface_embedding():
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embeddings = huggingface_embedding()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not PINECONE_API_KEY :
    raise ValueError("PINECONE_API_KEY is not set in .env!")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in .env!")



pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)
index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)


vector_store = PineconeVectorStore.from_documents(
    documents = text_chunks,
    index_name = index_name,
    embedding=embeddings
)

retriever = vector_store.as_retriever(
    search_type = "similarity",
   search_kwargs = {"k":4}
)