import json
from pathlib import Path

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT.joinpath("data", "movies.json")


def load_movies() -> list[dict]:
    movies_data = None
    with open(DATA_PATH, "r") as f:
        movies_data = json.load(f)

    return movies_data["movies"]
