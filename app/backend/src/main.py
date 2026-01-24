from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
def chat(payload: dict):
    # return your real response here
    # must return JSON like: {"answer": "...", "sources": [...]}
    return {"answer": "Hello from backend", "sources": []}
