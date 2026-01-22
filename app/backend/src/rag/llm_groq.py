from groq import Groq
from src.core.config import GROQ_API_KEY, GROQ_MODEL


def answer_with_groq(question: str, context: str) -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY missing. Put it in your .env")

    client = Groq(api_key=GROQ_API_KEY)

    system = (
        "You are PersonaQuery, a professional assistant.\n"
        "Answer ONLY using the provided context.\n"
        "If the answer is not supported by the context, say exactly:\n"
        "\"I don't have enough evidence in the provided documents.\"\n"
        "Do not invent facts.\n"
        "Keep the answer concise and interview-ready.\n"
    )

    user = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nReturn the best possible answer using only context."

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()
