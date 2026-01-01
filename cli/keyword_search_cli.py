#!/usr/bin/env python3

import argparse
from lib.keyword_search import (
    search_command,
    build_command,
    tf_command,
    idf_command,
    tf_idf_command,
    bm25_idf_command,
    bm25_tf_command,
    bm25_search_command,
)
from lib.search_utils import BM25_K1, BM25_B


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Builds the inverted index for the movies")

    tf_parser = subparsers.add_parser(
        "tf",
        help="Returns the times the token appears in the document with the given ID",
    )
    tf_parser.add_argument("doc_id", type=int, help="The id of the document")
    tf_parser.add_argument(
        "term", type=str, help="The term to return the frequency for"
    )

    idf_parser = subparsers.add_parser(
        "idf", help="Calculates the IDF for the given term"
    )
    idf_parser.add_argument("term", type=str, help="The word to calculate the IDF for")

    tfidf_parser = subparsers.add_parser(
        "tfidf",
        help="Returns the TF-IDF scoret of the provided term from the provided doc id",
    )
    tfidf_parser.add_argument("doc_id", type=int, help="The id of the document")
    tfidf_parser.add_argument(
        "term", type=str, help="The term to return the frequency for"
    )

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Calculates the BM25 IDF score for the given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="The word to calculate the BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )
    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            # Searching for: QUERY
            results = search_command(args.query)
            if results:
                print(f"Searching for: {args.query}")
                for i, res in enumerate(results, 1):
                    print(f"{i}. {res['title']}")
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "tf":
            count = tf_command(args.doc_id, args.term)
            print(f"{count}")
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tfidf = tf_idf_command(args.doc_id, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {
                    tfidf:.2f}"
            )
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {
                    bm25tf:.2f}"
            )
        case "bm25search":
            results = bm25_search_command(args.query)
            if results:
                for i, res in enumerate(results, 1):
                    print(
                        f"{i}. ({res['id']}) {res['title']} - Score: {res['score']:.2f}"
                    )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
