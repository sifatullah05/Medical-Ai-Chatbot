# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.chains import medical_chain

app = FastAPI(
    title="Medical Chatbot API",
    description="RAG-based মেডিকেল চ্যাটবট (LangChain, Pinecone & Groq ব্যবহার করে)",
    version="1.0"
)

# Request model
class ChatRequest(BaseModel):
    session_id: str
    question: str

# Root endpoint
@app.get("/")
def root():
    return {"message": "Medical Chatbot API চালু আছে 🚑"}

# Chat endpoint
@app.post("/chat")
def chat(request: ChatRequest):
    try:
        response = medical_chain.invoke({
            "question": request.question,
            "session_id": request.session_id
        })
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




