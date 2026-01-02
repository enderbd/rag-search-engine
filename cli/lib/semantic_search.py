from sentence_transformers import SentenceTransformer
import numpy as np
from .search_utils import CACHE_ROOT, load_movies


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

        self.embeddings_file = CACHE_ROOT.joinpath("movie_embeddings.npy")

    def generate_embedding(self, text: str) -> None:
        if not text or text.isspace():
            raise ValueError("Text is empty or made of whitespaces only!")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents):
        self.documents = documents
        to_embed = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            to_embed.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(to_embed, show_progress_bar=True)

        if not CACHE_ROOT.exists():
            CACHE_ROOT.mkdir()

        with open(self.embeddings_file, "wb") as f:
            np.save(f, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if self.embeddings_file.exists():
            with open(self.embeddings_file, "rb") as f:
                self.embeddings = np.load(f)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        else:
            return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)
        score_docs_pairs = []
        for i, embedding in enumerate(self.embeddings):
            score = cosine_similarity(query_embedding, embedding)
            doc = self.documents[i]
            score_docs_pairs.append((score, doc))
        score_docs_pairs.sort(key=lambda x: x[0], reverse=True)

        results = []
        for pair in score_docs_pairs[:limit]:
            movie_info = {
                "score": pair[0],
                "title": pair[1]["title"],
                "description": pair[1]["description"],
            }
            results.append(movie_info)

        return results


def verify_model() -> None:
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")


def embed_text(text: str) -> None:
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    movies = load_movies()
    search_instance = SemanticSearch()
    embeddings = search_instance.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {
            embeddings.shape[1]
        } dimensions"
    )


def embed_query_text(query):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def semantic_search_command(query, limit):
    movies = load_movies()
    search_instance = SemanticSearch()
    search_instance.load_or_create_embeddings(movies)
    results = search_instance.search(query, limit)
    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    print()
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score : {result['score']:.4f})")
        print(f"   {result['description'][:100]}...")
        print()


def fixed_size_chunking(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks = []

    n_words = len(words)
    i = 0
    while i < n_words:
        chunk_words = words[i : i + chunk_size]
        if chunks and len(chunk_words) <= overlap:
            break

        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap

    return chunks


def chunk_command(text: str, chunk_size: int, overlap: int) -> None:
    chunks = fixed_size_chunking(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
