"""
Base scraper interface for journal article fetching.
Defines the contract that all journal-specific scrapers must implement.
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import requests
from bs4 import BeautifulSoup
from loguru import logger

from config.settings import settings


@dataclass
class ArticleMetadata:
    """Metadata for a journal article."""

    title: str
    doi: str
    authors: List[str]
    publication_date: datetime
    abstract: str
    url: str
    journal: str
    issue: Optional[str] = None
    volume: Optional[str] = None
    keywords: Optional[List[str]] = None
    article_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["publication_date"] = self.publication_date.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArticleMetadata":
        """Create from dictionary."""
        data["publication_date"] = datetime.fromisoformat(data["publication_date"])
        return cls(**data)


@dataclass
class Article:
    """Complete article with metadata and full text."""

    metadata: ArticleMetadata
    fulltext: str
    sections: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": self.metadata.to_dict(),
            "fulltext": self.fulltext,
            "sections": self.sections or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Article":
        """Create from dictionary."""
        return cls(
            metadata=ArticleMetadata.from_dict(data["metadata"]),
            fulltext=data["fulltext"],
            sections=data.get("sections"),
        )


class BaseScraper(ABC):
    """Abstract base class for journal scrapers."""

    def __init__(self):
        """Initialize the scraper."""
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "OpenAccessRAG/1.0 (Research Tool; mailto:researcher@example.com)"
            }
        )

    @abstractmethod
    def get_journal_name(self) -> str:
        """Return the journal name."""
        pass

    @abstractmethod
    def fetch_issue_list(self, num_issues: int = 10) -> List[Dict[str, str]]:
        """
        Fetch a list of recent issues.

        Args:
            num_issues: Number of recent issues to fetch

        Returns:
            List of issue dictionaries with metadata (url, date, volume, etc.)
        """
        pass

    @abstractmethod
    def fetch_article_list(self, issue_url: str) -> List[str]:
        """
        Fetch list of article URLs from an issue.

        Args:
            issue_url: URL of the journal issue

        Returns:
            List of article URLs
        """
        pass

    @abstractmethod
    def fetch_article_metadata(self, article_url: str) -> ArticleMetadata:
        """
        Fetch metadata for a single article.

        Args:
            article_url: URL of the article

        Returns:
            ArticleMetadata object
        """
        pass

    @abstractmethod
    def fetch_article_fulltext(self, article_url: str) -> str:
        """
        Fetch the full text of an article.

        Args:
            article_url: URL of the article

        Returns:
            Full text content
        """
        pass

    def fetch_article(self, article_url: str) -> Article:
        """
        Fetch complete article (metadata + full text).

        Args:
            article_url: URL of the article

        Returns:
            Complete Article object
        """
        logger.info(f"Fetching article: {article_url}")

        metadata = self.fetch_article_metadata(article_url)
        fulltext = self.fetch_article_fulltext(article_url)

        return Article(metadata=metadata, fulltext=fulltext)

    def fetch_articles_from_issue(self, issue_url: str) -> List[Article]:
        """
        Fetch all articles from a single issue.

        Args:
            issue_url: URL of the journal issue

        Returns:
            List of Article objects
        """
        logger.info(f"Fetching articles from issue: {issue_url}")

        article_urls = self.fetch_article_list(issue_url)
        articles = []

        for url in article_urls:
            try:
                article = self.fetch_article(url)
                articles.append(article)
                self._rate_limit()
            except Exception as e:
                logger.error(f"Failed to fetch article {url}: {e}")
                continue

        logger.info(f"Successfully fetched {len(articles)}/{len(article_urls)} articles")
        return articles

    def fetch_recent_articles(self, num_issues: int = 10) -> List[Article]:
        """
        Fetch articles from recent issues.

        Args:
            num_issues: Number of recent issues to process

        Returns:
            List of all Article objects from the issues
        """
        logger.info(
            f"Fetching articles from {num_issues} recent issues of {self.get_journal_name()}"
        )

        issues = self.fetch_issue_list(num_issues)
        all_articles = []

        for issue in issues:
            try:
                articles = self.fetch_articles_from_issue(issue["url"])
                all_articles.extend(articles)
                self._rate_limit()
            except Exception as e:
                logger.error(f"Failed to fetch issue {issue.get('url')}: {e}")
                continue

        logger.info(
            f"Successfully fetched {len(all_articles)} total articles from {len(issues)} issues"
        )
        return all_articles

    def save_articles(
        self, articles: List[Article], output_dir: Optional[Path] = None
    ) -> None:
        """
        Save articles to disk as JSON files.

        Args:
            articles: List of articles to save
            output_dir: Directory to save articles (default: settings.raw_data_dir)
        """
        if output_dir is None:
            output_dir = settings.raw_data_dir / self.get_journal_name()

        output_dir.mkdir(parents=True, exist_ok=True)

        for article in articles:
            # Use DOI as filename (replace / with _)
            filename = article.metadata.doi.replace("/", "_") + ".json"
            filepath = output_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(article.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(articles)} articles to {output_dir}")

    def _make_request(
        self, url: str, max_retries: Optional[int] = None
    ) -> requests.Response:
        """
        Make HTTP request with retry logic.

        Args:
            url: URL to fetch
            max_retries: Maximum number of retries (default: settings.max_retries)

        Returns:
            Response object

        Raises:
            requests.RequestException: If all retries fail
        """
        if max_retries is None:
            max_retries = settings.max_retries

        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=settings.request_timeout)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                time.sleep(2**attempt)  # Exponential backoff

    def _rate_limit(self) -> None:
        """Apply rate limiting delay."""
        time.sleep(settings.rate_limit_delay)

    def _parse_html(self, html: str) -> BeautifulSoup:
        """Parse HTML content."""
        return BeautifulSoup(html, "lxml")
