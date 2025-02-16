from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API FastAPI!"}

@app.get("/hello/{name}")
def say_hello(name: str):
    return {"message": f"Bonjour {name}!"}
