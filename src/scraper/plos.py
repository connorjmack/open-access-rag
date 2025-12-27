"""
PLOS journal scraper implementation.
Fetches articles from PLOS journals (e.g., PLOS Climate).
"""

import re
from datetime import datetime
from typing import List, Dict, Optional

from bs4 import BeautifulSoup
from loguru import logger

from src.scraper.base import BaseScraper, ArticleMetadata


class PLOSScraper(BaseScraper):
    """Scraper for PLOS journals."""

    def __init__(self, journal_name: str = "climate"):
        """
        Initialize PLOS scraper.

        Args:
            journal_name: PLOS journal name (e.g., 'climate', 'one', 'biology')
        """
        super().__init__()
        self.journal_name = journal_name
        self.base_url = f"https://journals.plos.org/{journal_name}"

    def get_journal_name(self) -> str:
        """Return the journal name."""
        return f"plos-{self.journal_name}"

    def fetch_issue_list(self, num_issues: int = 10) -> List[Dict[str, str]]:
        """
        Fetch a list of recent issues.

        For PLOS, we'll fetch from the browse page.

        Args:
            num_issues: Number of recent issues to fetch

        Returns:
            List of issue dictionaries
        """
        # PLOS doesn't have traditional issues, so we'll fetch from browse/recent
        browse_url = f"{self.base_url}/browse"

        logger.info(f"Fetching issue list from {browse_url}")

        try:
            response = self._make_request(browse_url)
            soup = self._parse_html(response.text)

            # For PLOS, we'll treat the browse page as one "issue"
            # and just return a reference to fetch recent articles
            return [
                {
                    "url": browse_url,
                    "date": datetime.now().isoformat(),
                    "title": "Recent Articles",
                }
            ]

        except Exception as e:
            logger.error(f"Failed to fetch issue list: {e}")
            return []

    def fetch_article_list(self, issue_url: str) -> List[str]:
        """
        Fetch list of article URLs from the browse page.

        Args:
            issue_url: URL of the browse page

        Returns:
            List of article URLs
        """
        logger.info(f"Fetching article list from {issue_url}")

        try:
            response = self._make_request(issue_url)
            soup = self._parse_html(response.text)

            # Find all article links
            article_links = []

            # PLOS articles typically have links with /article?id= pattern
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if "/article?id=" in href or "/article/id/" in href:
                    # Construct full URL
                    if href.startswith("http"):
                        article_url = href
                    elif href.startswith("/"):
                        article_url = f"https://journals.plos.org{href}"
                    else:
                        article_url = f"{self.base_url}/{href}"

                    if article_url not in article_links:
                        article_links.append(article_url)

            logger.info(f"Found {len(article_links)} articles")
            return article_links

        except Exception as e:
            logger.error(f"Failed to fetch article list: {e}")
            return []

    def fetch_article_metadata(self, article_url: str) -> ArticleMetadata:
        """
        Fetch metadata for a single PLOS article.

        Args:
            article_url: URL of the article

        Returns:
            ArticleMetadata object
        """
        logger.debug(f"Fetching metadata for {article_url}")

        response = self._make_request(article_url)
        soup = self._parse_html(response.text)

        # Extract DOI
        doi = self._extract_doi(soup, article_url)

        # Extract title
        title = self._extract_title(soup)

        # Extract authors
        authors = self._extract_authors(soup)

        # Extract publication date
        pub_date = self._extract_publication_date(soup)

        # Extract abstract
        abstract = self._extract_abstract(soup)

        # Extract keywords
        keywords = self._extract_keywords(soup)

        # Extract article type
        article_type = self._extract_article_type(soup)

        return ArticleMetadata(
            title=title,
            doi=doi,
            authors=authors,
            publication_date=pub_date,
            abstract=abstract,
            url=article_url,
            journal=self.get_journal_name(),
            keywords=keywords,
            article_type=article_type,
        )

    def fetch_article_fulltext(self, article_url: str) -> str:
        """
        Fetch the full text of a PLOS article.

        Args:
            article_url: URL of the article

        Returns:
            Full text content
        """
        logger.debug(f"Fetching fulltext for {article_url}")

        response = self._make_request(article_url)
        soup = self._parse_html(response.text)

        # Find the article body
        article_body = soup.find("div", class_="article-content") or soup.find(
            "div", id="artText"
        )

        if not article_body:
            # Try to find main content
            article_body = soup.find("main") or soup.find("article")

        if article_body:
            # Remove script and style tags
            for tag in article_body.find_all(["script", "style"]):
                tag.decompose()

            # Extract text
            fulltext = article_body.get_text(separator="\n", strip=True)
            return fulltext

        logger.warning(f"Could not extract fulltext from {article_url}")
        return ""

    # Helper methods for metadata extraction

    def _extract_doi(self, soup: BeautifulSoup, url: str) -> str:
        """Extract DOI from article page."""
        # Try meta tag
        doi_meta = soup.find("meta", attrs={"name": "citation_doi"})
        if doi_meta and doi_meta.get("content"):
            return doi_meta["content"]

        # Try to extract from URL
        doi_match = re.search(r"id=(10\.\d+/[^\s&]+)", url)
        if doi_match:
            return doi_match.group(1)

        # Default fallback
        return f"unknown-doi-{datetime.now().timestamp()}"

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract article title."""
        # Try meta tag
        title_meta = soup.find("meta", attrs={"name": "citation_title"})
        if title_meta and title_meta.get("content"):
            return title_meta["content"]

        # Try h1 tag
        h1 = soup.find("h1", class_="title")
        if h1:
            return h1.get_text(strip=True)

        return "Unknown Title"

    def _extract_authors(self, soup: BeautifulSoup) -> List[str]:
        """Extract list of authors."""
        authors = []

        # Try meta tags
        author_metas = soup.find_all("meta", attrs={"name": "citation_author"})
        if author_metas:
            return [meta["content"] for meta in author_metas if meta.get("content")]

        # Try author list in HTML
        author_list = soup.find("ul", class_="author-list") or soup.find(
            "div", class_="authors"
        )
        if author_list:
            for author in author_list.find_all("li") or author_list.find_all("span"):
                author_name = author.get_text(strip=True)
                if author_name:
                    authors.append(author_name)

        return authors if authors else ["Unknown Author"]

    def _extract_publication_date(self, soup: BeautifulSoup) -> datetime:
        """Extract publication date."""
        # Try meta tag
        date_meta = soup.find("meta", attrs={"name": "citation_publication_date"})
        if date_meta and date_meta.get("content"):
            try:
                return datetime.strptime(date_meta["content"], "%Y/%m/%d")
            except ValueError:
                pass

        # Try different date formats
        date_meta = soup.find("meta", attrs={"name": "date"})
        if date_meta and date_meta.get("content"):
            try:
                return datetime.fromisoformat(date_meta["content"])
            except ValueError:
                pass

        # Default to current date
        return datetime.now()

    def _extract_abstract(self, soup: BeautifulSoup) -> str:
        """Extract article abstract."""
        # Try meta tag
        abstract_meta = soup.find("meta", attrs={"name": "citation_abstract"})
        if abstract_meta and abstract_meta.get("content"):
            return abstract_meta["content"]

        # Try abstract section
        abstract_section = soup.find("div", class_="abstract") or soup.find(
            "section", id="abstract"
        )
        if abstract_section:
            return abstract_section.get_text(separator=" ", strip=True)

        return ""

    def _extract_keywords(self, soup: BeautifulSoup) -> Optional[List[str]]:
        """Extract article keywords."""
        keywords = []

        # Try meta tags
        keyword_meta = soup.find("meta", attrs={"name": "citation_keywords"})
        if keyword_meta and keyword_meta.get("content"):
            keywords = [kw.strip() for kw in keyword_meta["content"].split(",")]

        # Try keyword section
        if not keywords:
            keyword_section = soup.find("div", class_="keywords") or soup.find(
                "section", class_="keywords"
            )
            if keyword_section:
                keywords = [
                    kw.get_text(strip=True)
                    for kw in keyword_section.find_all("a") or keyword_section.find_all("span")
                ]

        return keywords if keywords else None

    def _extract_article_type(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract article type/category."""
        # Try meta tag
        type_meta = soup.find("meta", attrs={"name": "citation_article_type"})
        if type_meta and type_meta.get("content"):
            return type_meta["content"]

        # Try article type section
        type_section = soup.find("p", class_="article-type")
        if type_section:
            return type_section.get_text(strip=True)

        return None
