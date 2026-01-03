#!/usr/bin/env python3

import argparse

from lib.search_utils import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
)
from lib.semantic_search import (
    chunk_command,
    embed_chunks_command,
    embed_query_text,
    embed_text,
    search_chunked_command,
    semantic_chunk_command,
    semantic_search_command,
    verify_embeddings,
    verify_model,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verifies the semantic search model")

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="embeds the provided text"
    )
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser(
        "verify_embeddings", help="Verifies the correct embeddings for the movies"
    )

    embed_query_parser = subparsers.add_parser(
        "embedquery", help="embeds the provided text"
    )
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser(
        "search",
        help="Search for a movie using the provided query with an optional limit",
    )
    search_parser.add_argument("query", type=str, help="QUery to use")
    search_parser.add_argument(
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="The search limit"
    )

    chunck_parser = subparsers.add_parser("chunk", help="Splits a text in N chunks")
    chunck_parser.add_argument("text", type=str, help="Text to chunk")
    chunck_parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Lenght of chunk"
    )
    chunck_parser.add_argument(
        "--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Overlap size"
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Semantic chunking"
    )

    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, default=4, help="Lenght of chunk"
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Overlap size"
    )

    subparsers.add_parser("embed_chunks", help="Builds the chunks embeddings!")

    search_chunked_parser = subparsers.add_parser(
        "search_chunked",
        help="Search for a movie using the provided query with an optional limit",
    )
    search_chunked_parser.add_argument("query", type=str, help="QUery to use")
    search_chunked_parser.add_argument(
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="The search limit"
    )

    args = parser.parse_args()
    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            semantic_search_command(args.query, args.limit)
        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks_command()
        case "search_chunked":
            search_chunked_command(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
