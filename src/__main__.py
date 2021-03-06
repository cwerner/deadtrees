import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("command")
parser.add_argument("args", nargs="*")


def main():
    args = parser.parse_args()

    if args.command == "version":
        from src.version import __version__

        print(__version__)
        sys.exit(0)

    # if args.command == "search":
    #     from mtg_search.api import search

    #     query = " ".join(args.args)
    #     print(f"query is: {query}")
    #     print()
    #     for c in search(query):
    #         print(c.name)
    #         print(c.text)
    #         print()
    #     sys.exit(0)

    # if args.command == "index":
    #     from mtg_search.api import create_index

    #     create_index()
    #     sys.exit(0)

    # if args.command == "checkpoint":
    #     from mtg_search.checkpoint import CHECKPOINT

    #     print(CHECKPOINT)
    #     sys.exit(0)

    # else:
    #     print("unknown command")
    #     sys.exit(1)


if __name__ == "__main__":
    main()