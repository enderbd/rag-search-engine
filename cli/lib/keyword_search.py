import pickle
import string

from nltk.stem import PorterStemmer

from .search_utils import CACHE_ROOT, DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}

    def __add_document(self, doc_id, text):
        document_tokens = tokenize_text(text)
        for word in document_tokens:
            if word not in self.index:
                self.index[word] = set()
            self.index[word].add(doc_id)

    def get_documents(self, term):
        if term.lower() in self.index.keys():
            return sorted(list(self.index[term.lower()]))
        return []

    def build(self, movies):
        for movie in movies:
            doc_id = movie["id"]
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie

    def save(self):
        cache_folder = CACHE_ROOT
        if not cache_folder.is_dir():
            cache_folder.mkdir()

        index_file = cache_folder.joinpath("index.pkl")
        docmap_file = cache_folder.joinpath("docmap.pkl")

        with open(index_file, "wb") as f:
            pickle.dump(self.index, f)

        with open(docmap_file, "wb") as f:
            pickle.dump(self.docmap, f)


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    if movies:
        for movie in movies:
            query_tokens = tokenize_text(query)
            title_tokens = tokenize_text(movie["title"])
            if has_matching_token(query_tokens, title_tokens):
                results.append(movie)
            if len(results) >= limit:
                break
    return results


def build_command() -> None:
    movies = load_movies()
    idx = InvertedIndex()
    idx.build(movies)
    idx.save()


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)

    stop_words = load_stopwords()
    filtered_words = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)

    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))

    return stemmed_words
