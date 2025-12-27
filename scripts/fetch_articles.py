#!/usr/bin/env python3
"""
CLI script to fetch articles from open access journals.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from loguru import logger

from src.scraper.plos import PLOSScraper
from config.settings import settings


@click.command()
@click.option(
    "--journal",
    "-j",
    default="climate",
    help="PLOS journal name (e.g., 'climate', 'one', 'biology')",
)
@click.option(
    "--num-issues",
    "-n",
    default=None,
    type=int,
    help=f"Number of recent issues to fetch (default: {settings.num_issues})",
)
@click.option(
    "--output-dir",
    "-o",
    default=None,
    help="Output directory for articles (default: data/raw/<journal>)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def main(journal, num_issues, output_dir, verbose):
    """
    Fetch articles from PLOS journals.

    Example:
        python scripts/fetch_articles.py --journal climate --num-issues 10
    """
    # Configure logging
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    # Use settings default if not specified
    if num_issues is None:
        num_issues = settings.num_issues

    # Ensure directories exist
    settings.ensure_directories()

    logger.info(f"Fetching articles from PLOS {journal.upper()}")
    logger.info(f"Target: {num_issues} recent issues")

    try:
        # Initialize scraper
        scraper = PLOSScraper(journal_name=journal)

        # Fetch articles
        articles = scraper.fetch_recent_articles(num_issues=num_issues)

        if not articles:
            logger.error("No articles fetched. Please check the journal name and try again.")
            sys.exit(1)

        # Save articles
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = None

        scraper.save_articles(articles, output_dir=output_path)

        logger.success(f"Successfully fetched and saved {len(articles)} articles")

        # Print summary
        print("\n" + "=" * 60)
        print("FETCH SUMMARY")
        print("=" * 60)
        print(f"Journal: PLOS {journal.upper()}")
        print(f"Articles fetched: {len(articles)}")
        print(f"Output directory: {output_path or settings.raw_data_dir / scraper.get_journal_name()}")
        print("=" * 60 + "\n")

        print("Next steps:")
        print("1. Process the articles: python scripts/process_corpus.py")
        print("2. Launch the UI: streamlit run src/ui/app.py")

    except KeyboardInterrupt:
        logger.warning("\nFetch interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to fetch articles: {e}")
        if verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
