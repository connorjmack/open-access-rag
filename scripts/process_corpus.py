#!/usr/bin/env python3
"""
CLI script to process articles and generate embeddings.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from loguru import logger
from tqdm import tqdm

from src.scraper.base import Article
from src.processor.text_processor import process_article_text
from src.processor.embeddings import EmbeddingGenerator
from src.storage.vector_store import VectorStore
from config.settings import settings


@click.command()
@click.option(
    "--input-dir",
    "-i",
    default=None,
    help="Input directory with article JSON files (default: data/raw)",
)
@click.option(
    "--clear-existing",
    "-c",
    is_flag=True,
    help="Clear existing vector store before processing",
)
@click.option(
    "--batch-size",
    "-b",
    default=None,
    type=int,
    help=f"Batch size for embedding generation (default: {settings.embedding_batch_size})",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def main(input_dir, clear_existing, batch_size, verbose):
    """
    Process articles and generate embeddings for the vector store.

    This script:
    1. Loads article JSON files
    2. Processes text into chunks
    3. Generates embeddings
    4. Stores in ChromaDB

    Example:
        python scripts/process_corpus.py --input-dir data/raw/plos-climate
    """
    # Configure logging
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    # Ensure directories exist
    settings.ensure_directories()

    # Determine input directory
    if input_dir:
        input_path = Path(input_dir)
    else:
        input_path = settings.raw_data_dir

    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        sys.exit(1)

    logger.info(f"Processing articles from: {input_path}")

    try:
        # Initialize components
        logger.info("Initializing components...")
        vector_store = VectorStore()
        embedding_generator = EmbeddingGenerator(
            batch_size=batch_size or settings.embedding_batch_size
        )

        # Clear existing data if requested
        if clear_existing:
            logger.warning("Clearing existing vector store...")
            vector_store.clear()

        # Find all JSON files
        json_files = list(input_path.rglob("*.json"))

        if not json_files:
            logger.error(f"No JSON files found in {input_path}")
            sys.exit(1)

        logger.info(f"Found {len(json_files)} article files")

        # Process articles
        all_chunks = []
        all_embeddings = []
        processed_count = 0

        for json_file in tqdm(json_files, desc="Processing articles"):
            try:
                # Load article
                with open(json_file, "r", encoding="utf-8") as f:
                    article_data = json.load(f)

                article = Article.from_dict(article_data)

                # Create metadata dict for chunks
                chunk_metadata = {
                    "title": article.metadata.title,
                    "doi": article.metadata.doi,
                    "authors": ", ".join(article.metadata.authors),
                    "publication_date": article.metadata.publication_date.isoformat(),
                    "journal": article.metadata.journal,
                    "url": article.metadata.url,
                }

                if article.metadata.keywords:
                    chunk_metadata["keywords"] = ", ".join(article.metadata.keywords)

                if article.metadata.article_type:
                    chunk_metadata["article_type"] = article.metadata.article_type

                # Process text into chunks
                chunks = process_article_text(
                    fulltext=article.fulltext,
                    metadata=chunk_metadata,
                    use_sections=True,
                )

                if not chunks:
                    logger.warning(f"No chunks generated for {article.metadata.doi}")
                    continue

                # Generate embeddings for chunks
                chunk_texts = [chunk.text for chunk in chunks]
                embeddings = embedding_generator.embed_texts(
                    chunk_texts, show_progress=False
                )

                # Add to vector store
                vector_store.add_chunks(
                    chunks=chunks,
                    embeddings=embeddings,
                    article_metadata=chunk_metadata,
                )

                processed_count += 1

            except Exception as e:
                logger.error(f"Failed to process {json_file.name}: {e}")
                if verbose:
                    logger.exception("Full traceback:")
                continue

        logger.success(f"Successfully processed {processed_count}/{len(json_files)} articles")

        # Print summary
        stats = vector_store.get_stats()

        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Articles processed: {processed_count}")
        print(f"Total chunks in store: {stats['total_chunks']}")
        print(f"Unique articles: {stats['unique_articles']}")
        print(f"Journals: {', '.join(stats['journals'])}")

        if stats['date_range'][0]:
            print(f"Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")

        print("=" * 60 + "\n")

        print("Next step:")
        print("Launch the UI: streamlit run src/ui/app.py")

    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to process corpus: {e}")
        if verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
