
from fastapi import FastAPI
from pydantic import BaseModel
from rag import process_message  # 🔥 Import de la fonction de ton IA

app = FastAPI()

class ChatInput(BaseModel):
    message: str

@app.post("/chat")
def chat_endpoint(input: ChatInput):
    # Appelle la fonction d'IA pour analyser le message du patient
    response = process_message(input.message)
    return {"reply": response}
