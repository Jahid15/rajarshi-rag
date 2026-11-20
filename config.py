import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BOOK_JSON = DATA_DIR / "book.json"
INDEX_ROOT = ROOT / "index"
RESULTS_DIR = ROOT / "results"

BOOK_TITLE = "রাজর্ষি (১৯৬১)"
BOOK_AUTHOR = "রবীন্দ্রনাথ ঠাকুর"
WIKI_BASE = "https://bn.wikisource.org/wiki/"
API_URL = "https://bn.wikisource.org/w/api.php"
USER_AGENT = "BanglaBookRAG/1.0 (student coursework)"

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.45"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "")

DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "anthropic": "claude-opus-5-5",
}


def index_dir(model: str = EMBED_MODEL, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> Path:
    slug = model.split("/")[-1].replace(".", "_")
    return INDEX_ROOT / f"{slug}__{size}_{overlap}"


def page_url(title: str) -> str:
    return WIKI_BASE + title.replace(" ", "_")
