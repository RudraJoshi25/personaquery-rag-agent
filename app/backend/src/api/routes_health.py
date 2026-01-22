from fastapi import APIRouter
from src.core.config import AUTHOR_LINKS


router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok", "author_links": AUTHOR_LINKS}
