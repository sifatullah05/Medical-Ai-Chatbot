from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.embeddings import hugging_face_embeddings, text_split
from src.loader import minimal_docs
import os

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in .env!")

pinecone_api_key = PINECONE_API_KEY

pc = Pinecone(api_key = pinecone_api_key)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name= index_name,
        dimension=384,
        metric= "cosine",
        spec= ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)


vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=hugging_face_embeddings()
)

stats = vectorstore.index.describe_index_stats()

vector_count = (
    stats.get("namespaces", {})
         .get("", {})
         .get("vector_count", 0)
)

if vector_count == 0:
    vectorstore = PineconeVectorStore.from_documents(
        documents=text_split(minimal_docs),
        index_name=index_name,
        embedding=hugging_face_embeddings()
    )

retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":3}
)