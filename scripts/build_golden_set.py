#!/usr/bin/env python3
import argparse
from pathlib import Path
from src.ingestion.loader import DocumentLoader, DocumentChunker, ChunkingStrategy
from src.evaluation.golden_set_builder import GoldenSetBuilder


def main():
    parser = argparse.ArgumentParser(description="Build golden Q&A set from documents")
    parser.add_argument("--input", required=True, help="Input directory containing documents")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--n", type=int, default=30, help="Number of questions to generate")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model to use")

    args = parser.parse_args()

    # Load documents from input directory
    input_path = Path(args.input)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_path}")

    loader = DocumentLoader()
    chunker = DocumentChunker()
    all_chunks = []

    # Iterate over all files in the input directory
    for file_path in input_path.iterdir():
        if file_path.is_file():
            try:
                docs = loader.load(str(file_path))
                chunks = chunker.chunk(docs, ChunkingStrategy.FIXED)
                all_chunks.extend(chunks)
                print(f"Processed {file_path}: {len(chunks)} chunks")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    if not all_chunks:
        print("No chunks loaded. Exiting.")
        return

    # Generate golden Q&A set
    builder = GoldenSetBuilder()
    golden_set = builder.generate_from_chunks(all_chunks, n_questions=args.n, llm_model=args.model)

    # Save to output file
    builder.save(golden_set, args.output)
    print(f"Successfully saved {len(golden_set)} golden Q&A pairs to {args.output}")


if __name__ == "__main__":
    main()
