import math
import pickle
import string
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer

from .search_utils import (
    BM25_B,
    BM25_K1,
    CACHE_ROOT,
    DEFAULT_SEARCH_LIMIT,
    format_search_result,
    load_movies,
    load_stopwords,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)
        self.doc_lengths = defaultdict()

        self.index_file = CACHE_ROOT.joinpath("index.pkl")
        self.docmap_file = CACHE_ROOT.joinpath("docmap.pkl")
        self.tf_file = CACHE_ROOT.joinpath("term_frequencies.pkl")
        self.doc_lengths_file = CACHE_ROOT.joinpath("doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        document_tokens = tokenize_text(text)
        for word in document_tokens:
            if word not in self.index:
                self.index[word] = set()
            self.index[word].add(doc_id)

        self.term_frequencies[doc_id].update(document_tokens)
        self.doc_lengths[doc_id] = len(document_tokens)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term.lower(), set())
        if term.lower() in self.index.keys():
            return sorted(list(doc_ids))
        return []

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("The term should be one word only")
        return self.term_frequencies[doc_id][tokens[0]]

    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("The term should be one word only")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])

        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        return self.get_idf(term) * self.get_tf(doc_id, term)

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("The term should be one word only")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])

        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        length_norm = 1
        if self.__get_avg_doc_length() != 0:
            length_norm = (
                1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
            )

        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def build(self, movies: list[dict]) -> None:
        for movie in movies:
            doc_id = movie["id"]
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie

    def save(self) -> None:
        if not CACHE_ROOT.is_dir():
            CACHE_ROOT.mkdir()

        with open(self.index_file, "wb") as f:
            pickle.dump(self.index, f)

        with open(self.docmap_file, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(self.tf_file, "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(self.doc_lengths_file, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        if not self.index_file.exists():
            raise FileNotFoundError(
                "Could not find a saved file for the index, you need to build first"
            )
        if not self.docmap_file.exists():
            raise FileNotFoundError(
                "Could not find a saved file for the docmap, you need to build first"
            )

        if not self.tf_file.exists():
            raise FileNotFoundError(
                "Could not find a saved file for the term_frequencies, you need to build first"
            )

        if not self.doc_lengths_file.exists():
            raise FileNotFoundError(
                "Could not find a saved file for the doc lenghts, you need to build first"
            )

        with open(self.index_file, "rb") as f:
            self.index = pickle.load(f)

        with open(self.docmap_file, "rb") as f:
            self.docmap = pickle.load(f)

        with open(self.tf_file, "rb") as f:
            self.term_frequencies = pickle.load(f)

        with open(self.doc_lengths_file, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0

        return sum(list(self.doc_lengths.values())) / len(self.doc_lengths)

    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        scores = {}
        query_tokens = tokenize_text(query)
        for doc_id in self.docmap:
            score = 0.0
            for token in query_tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in sorted_docs[:limit]:
            doc = self.docmap[doc_id]
            formatted_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"],
                score=score,
            )
            results.append(formatted_result)

        return results


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Could not find saved index files. Please run the build command first.")
        exit(1)

    id_results = []

    query_tokens = tokenize_text(query)
    for token in query_tokens:
        docs = idx.get_documents(token)
        for doc_id in docs:
            if doc_id in id_results:
                continue
            id_results.append(doc_id)
            if len(id_results) >= limit:
                break

        if len(id_results) >= limit:
            break

    results = []
    for id in id_results[:limit]:
        results.append(idx.docmap[id])
    return results


def bm25_search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Could not find saved index files. Please run the build command first.")
        exit(1)

    return idx.bm25_search(query, limit)


def build_command() -> None:
    movies = load_movies()
    idx = InvertedIndex()
    idx.build(movies)
    idx.save()


def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Could not find saved  files. Please run the build command first.")
        exit(1)
    return idx.get_tf(doc_id, term)


def idf_command(term: str) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Could not find saved  files. Please run the build command first.")
        exit(1)
    return idx.get_idf(term)


def tf_idf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Could not find saved  files. Please run the build command first.")
        exit(1)
    return idx.get_tf_idf(doc_id, term)


def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Could not find saved  files. Please run the build command first.")
        exit(1)
    return idx.get_bm25_idf(term)


def bm25_tf_command(
    doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Could not find saved  files. Please run the build command first.")
        exit(1)
    return idx.get_bm25_tf(doc_id, term, k1, b)


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
