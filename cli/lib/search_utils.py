import json
from pathlib import Path

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT.joinpath("data", "movies.json")
STOPWORDS_PATH = PROJECT_ROOT.joinpath("data", "stopwords.txt")
CACHE_ROOT = PROJECT_ROOT.joinpath("cache")


def load_movies() -> list[dict]:
    movies_data = None
    with open(DATA_PATH, "r") as f:
        movies_data = json.load(f)

    return movies_data["movies"]


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        stopwords_data = f.read()

    return stopwords_data.splitlines()
