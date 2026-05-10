import argparse
import sys
from gridglyph.generators.build import build_dataset

def main():
    parser = argparse.ArgumentParser(prog="gridglyph", description="GridGlyph Logic Engine")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: build-dataset
    parser_build = subparsers.add_parser("build-dataset", help="Generate the neuro-symbolic dataset")
    parser_build.add_argument("--output", type=str, default="atomic_dataset.jsonl", help="Output file path")
    parser_build.add_argument("--multiplier", type=int, default=1, help="Volume multiplier")

    args = parser.parse_args()

    if args.command == "build-dataset":
        build_dataset(output_path=args.output, multiplier=args.multiplier)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()