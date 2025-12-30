from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    if movies:
        for movie in movies:
            preprocessed_query = preprocess_text(query)
            preprocessed_title = preprocess_text(movie["title"])
            if preprocessed_query in preprocessed_title:
                results.append(movie)
            if len(results) >= limit:
                break
    return results


def preprocess_text(text: str) -> str:
    text = text.lower()
    return text
