"""
Corpus summarization using Claude.
"""

from typing import List, Dict, Any

import anthropic
from loguru import logger

from config.settings import settings


class CorpusSummarizer:
    """Generates summaries and insights from article corpus."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize summarizer.

        Args:
            api_key: Anthropic API key (defaults to settings)
            model: Claude model to use (defaults to settings)
        """
        from typing import Optional

        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or settings.llm_model
        self.client = anthropic.Anthropic(api_key=self.api_key)

        logger.info(f"Initialized CorpusSummarizer with model: {self.model}")

    def summarize_articles(
        self, articles_metadata: List[Dict[str, Any]], max_articles: int = 50
    ) -> str:
        """
        Generate a summary of the article corpus.

        Args:
            articles_metadata: List of article metadata dictionaries
            max_articles: Maximum number of articles to include in summary

        Returns:
            Summary text
        """
        if not articles_metadata:
            return "No articles available to summarize."

        # Limit number of articles to avoid token limits
        articles_subset = articles_metadata[:max_articles]

        # Build context from article metadata
        context = self._build_article_context(articles_subset)

        # Create prompt
        prompt = f"""Analyze the following collection of academic articles and provide a comprehensive summary.

{context}

Please provide:
1. An overview of the main research themes and topics
2. Key findings and contributions across the corpus
3. Notable trends in publication topics over time (if apparent from dates)
4. Common methodologies or approaches
5. Gaps or emerging areas of research

Keep the summary concise but informative (300-500 words)."""

        # Call Claude
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )

            summary = response.content[0].text
            logger.info("Generated corpus summary")
            return summary

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return f"Error generating summary: {str(e)}"

    def extract_topics(
        self, articles_metadata: List[Dict[str, Any]], num_topics: int = 10
    ) -> List[str]:
        """
        Extract main topics from article corpus.

        Args:
            articles_metadata: List of article metadata dictionaries
            num_topics: Number of topics to extract

        Returns:
            List of topic strings
        """
        if not articles_metadata:
            return []

        # Build context
        titles_and_abstracts = []
        for article in articles_metadata[:100]:  # Limit for token size
            title = article.get("title", "")
            abstract = article.get("abstract", "")
            if title and abstract:
                titles_and_abstracts.append(f"Title: {title}\nAbstract: {abstract}\n")

        context = "\n---\n".join(titles_and_abstracts)

        prompt = f"""Analyze the following academic articles and identify the {num_topics} main research topics or themes.

{context}

List exactly {num_topics} distinct topics, one per line. Each topic should be a concise phrase (2-5 words).
Format: Just list the topics, no numbering or bullet points."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )

            topics_text = response.content[0].text
            topics = [
                line.strip()
                for line in topics_text.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]

            logger.info(f"Extracted {len(topics)} topics")
            return topics[:num_topics]

        except Exception as e:
            logger.error(f"Failed to extract topics: {e}")
            return []

    def extract_keywords(
        self, articles_metadata: List[Dict[str, Any]], num_keywords: int = 20
    ) -> Dict[str, int]:
        """
        Extract and count keywords from articles.

        Args:
            articles_metadata: List of article metadata dictionaries
            num_keywords: Number of top keywords to return

        Returns:
            Dictionary mapping keywords to frequency counts
        """
        from collections import Counter

        # Collect all keywords from metadata
        all_keywords = []

        for article in articles_metadata:
            keywords = article.get("keywords", [])
            if keywords:
                all_keywords.extend([kw.lower().strip() for kw in keywords])

        # Count frequencies
        keyword_counts = Counter(all_keywords)

        # Return top N
        top_keywords = dict(keyword_counts.most_common(num_keywords))

        logger.info(f"Extracted {len(top_keywords)} top keywords")
        return top_keywords

    def analyze_trends(
        self, articles_metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze publication trends over time.

        Args:
            articles_metadata: List of article metadata dictionaries

        Returns:
            Dictionary with trend analysis
        """
        from collections import defaultdict
        from datetime import datetime

        # Group by year and month
        by_year = defaultdict(int)
        by_month = defaultdict(int)
        by_type = defaultdict(int)

        for article in articles_metadata:
            pub_date = article.get("publication_date")
            if pub_date:
                if isinstance(pub_date, str):
                    try:
                        pub_date = datetime.fromisoformat(pub_date)
                    except:
                        continue

                year = pub_date.year
                month = pub_date.strftime("%Y-%m")

                by_year[year] += 1
                by_month[month] += 1

            article_type = article.get("article_type")
            if article_type:
                by_type[article_type] += 1

        return {
            "by_year": dict(by_year),
            "by_month": dict(by_month),
            "by_type": dict(by_type),
            "total_articles": len(articles_metadata),
        }

    def _build_article_context(self, articles: List[Dict[str, Any]]) -> str:
        """Build formatted context from article metadata."""
        context_parts = []

        for i, article in enumerate(articles, 1):
            title = article.get("title", "Unknown")
            authors = ", ".join(article.get("authors", [])[:3])
            date = article.get("publication_date", "Unknown date")
            abstract = article.get("abstract", "")[:200]  # Truncate

            context_parts.append(
                f"Article {i}:\nTitle: {title}\nAuthors: {authors}\nDate: {date}\nAbstract: {abstract}...\n"
            )

        return "\n".join(context_parts)
