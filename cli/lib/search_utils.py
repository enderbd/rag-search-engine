import json
from pathlib import Path
from typing import Any

DEFAULT_SEARCH_LIMIT = 5
SCORE_PRECISION = 3
DEFAULT_ALPHA = 0.5

BM25_K1 = 1.5
BM25_B = 0.75

RRF_K = 60

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT.joinpath("data", "movies.json")
STOPWORDS_PATH = PROJECT_ROOT.joinpath("data", "stopwords.txt")
CACHE_ROOT = PROJECT_ROOT.joinpath("cache")

DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 0
DEFAULT_SEMANTIC_CHUNK_SIZE = 4

DOCUMENT_PREVIEW_LENGTH = 100


def load_movies() -> list[dict]:
    movies_data = None
    with open(DATA_PATH, "r") as f:
        movies_data = json.load(f)

    return movies_data["movies"]


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        stopwords_data = f.read()

    return stopwords_data.splitlines()


def format_search_result(
    doc_id: str, title: str, document: str, score: float, **metadata: Any
) -> dict[str, Any]:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }
