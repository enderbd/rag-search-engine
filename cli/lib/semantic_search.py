from sentence_transformers import SentenceTransformer
import numpy as np
from .search_utils import (
    CACHE_ROOT,
    load_movies,
    format_search_result,
    DOCUMENT_PREVIEW_LENGTH,
)
import re
import json


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

        self.chunk_embeddings_file = CACHE_ROOT.joinpath("chunk_embeddings.npy")
        self.chunk_metadata_file = CACHE_ROOT.joinpath("chunk_metadata.json")

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        all_chunks = []
        metadata = []

        for i, doc in enumerate(self.documents):
            self.document_map[doc["id"]] = doc
            if not doc["description"]:
                continue
            desc_chunks = semantic_chunking(doc["description"], 4, 1)
            all_chunks.extend(desc_chunks)
            for j, chunk in enumerate(desc_chunks):
                metadata.append(
                    {"movie_idx": i, "chunk_idx": j, "total_chunks": len(desc_chunks)}
                )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = metadata

        if not CACHE_ROOT.exists():
            CACHE_ROOT.mkdir()

        with open(self.chunk_embeddings_file, "wb") as f:
            np.save(f, self.chunk_embeddings)

        with open(self.chunk_metadata_file, "w") as f:
            json.dump(
                {"chunks": metadata, "total_chunks": len(all_chunks)}, f, indent=2
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if self.chunk_embeddings_file.exists() and self.chunk_metadata_file.exists():
            with open(self.chunk_embeddings_file, "rb") as f:
                self.chunk_embeddings = np.load(f)

            with open(self.chunk_metadata_file, "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]

            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int) -> list[dict]:
        query_embedding = self.generate_embedding(query)
        chunk_score = []
        for i, embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, embedding)
            chunk_idx = self.chunk_metadata[i]["chunk_idx"]
            movie_idx = self.chunk_metadata[i]["movie_idx"]
            chunk_score.append(
                {"chunk_idx": chunk_idx, "movie_idx": movie_idx, "score": score}
            )

        movie_idx_to_score = {}
        for chunk in chunk_score:
            movie_idx = chunk["movie_idx"]
            score = chunk["score"]
            if (
                movie_idx not in movie_idx_to_score
                or score > movie_idx_to_score[movie_idx]
            ):
                movie_idx_to_score[movie_idx] = score

        sorted_movies = sorted(
            movie_idx_to_score.items(), key=lambda x: x[1], reverse=True
        )

        results = []
        for movie_idx, score in sorted_movies[:limit]:
            doc = self.documents[movie_idx]
            formatted_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"][:DOCUMENT_PREVIEW_LENGTH],
                score=score,
            )
            results.append(formatted_result)

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


def semantic_chunking(text: str, max_chunk_size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1 and not text.endswith((".", "!", "?")):
        sentences = [text]

    chunks = []
    n_sentences = len(sentences)
    i = 0
    while i < n_sentences:
        chunk_sentences = sentences[i : i + max_chunk_size]
        cleaned_sentences = [s.strip() for s in chunk_sentences if s.strip()]

        if chunks and len(chunk_sentences) <= overlap:
            break

        i += max_chunk_size - overlap

        if not cleaned_sentences:
            continue
        chunks.append(" ".join(cleaned_sentences))

    return chunks


def semantic_chunk_command(text: str, max_chunk_size: int, overlap: int) -> list[str]:
    chunks = semantic_chunking(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")


def embed_chunks_command():
    movies = load_movies()
    chunk_semantic_instance = ChunkedSemanticSearch()
    embeddings = chunk_semantic_instance.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")


def search_chunked_command(query: str, limit: int) -> None:
    movies = load_movies()
    chunk_semantic_instance = ChunkedSemanticSearch()
    chunk_semantic_instance.load_or_create_chunk_embeddings(movies)
    results = chunk_semantic_instance.search_chunks(query, limit)

    for i, res in enumerate(results, 1):
        print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
        print(f"   {res['document']}...")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
