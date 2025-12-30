#!/usr/bin/env python3

import argparse
from lib.keyword_search import search_command, build_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Builds the inverted index for the movies")
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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
