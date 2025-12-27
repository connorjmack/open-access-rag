"""
Topic modeling and analysis.
"""

from collections import defaultdict, Counter
from datetime import datetime
from typing import List, Dict, Any, Tuple

from loguru import logger


class TopicAnalyzer:
    """Analyzes topics and trends in article corpus."""

    def __init__(self):
        """Initialize topic analyzer."""
        pass

    def analyze_keyword_trends(
        self, articles_metadata: List[Dict[str, Any]]
    ) -> Dict[str, List[Tuple[str, int]]]:
        """
        Analyze how keywords trend over time.

        Args:
            articles_metadata: List of article metadata dictionaries

        Returns:
            Dictionary mapping time periods to keyword frequencies
        """
        # Group keywords by year-month
        keywords_by_period = defaultdict(lambda: Counter())

        for article in articles_metadata:
            pub_date = article.get("publication_date")
            keywords = article.get("keywords", [])

            if pub_date and keywords:
                # Parse date
                if isinstance(pub_date, str):
                    try:
                        pub_date = datetime.fromisoformat(pub_date)
                    except:
                        continue

                period = pub_date.strftime("%Y-%m")

                # Count keywords for this period
                for keyword in keywords:
                    keywords_by_period[period][keyword.lower()] += 1

        # Convert to sorted lists
        trends = {}
        for period, counter in keywords_by_period.items():
            trends[period] = counter.most_common(10)  # Top 10 per period

        logger.info(f"Analyzed keyword trends across {len(trends)} time periods")
        return trends

    def get_topic_evolution(
        self, articles_metadata: List[Dict[str, Any]], topics: List[str]
    ) -> Dict[str, Dict[str, int]]:
        """
        Track how specific topics evolve over time.

        Args:
            articles_metadata: List of article metadata dictionaries
            topics: List of topics to track

        Returns:
            Dictionary mapping topics to time series data
        """
        topic_timeline = {topic: defaultdict(int) for topic in topics}

        for article in articles_metadata:
            pub_date = article.get("publication_date")
            title = article.get("title", "").lower()
            abstract = article.get("abstract", "").lower()
            keywords = [kw.lower() for kw in article.get("keywords", [])]

            if pub_date:
                # Parse date
                if isinstance(pub_date, str):
                    try:
                        pub_date = datetime.fromisoformat(pub_date)
                    except:
                        continue

                period = pub_date.strftime("%Y-%m")

                # Check which topics are mentioned
                for topic in topics:
                    topic_lower = topic.lower()

                    # Check if topic appears in title, abstract, or keywords
                    if (
                        topic_lower in title
                        or topic_lower in abstract
                        or any(topic_lower in kw for kw in keywords)
                    ):
                        topic_timeline[topic][period] += 1

        # Convert defaultdicts to regular dicts
        result = {topic: dict(timeline) for topic, timeline in topic_timeline.items()}

        logger.info(f"Tracked evolution of {len(topics)} topics")
        return result

    def get_publication_timeline(
        self, articles_metadata: List[Dict[str, Any]], granularity: str = "month"
    ) -> Dict[str, int]:
        """
        Get publication counts over time.

        Args:
            articles_metadata: List of article metadata dictionaries
            granularity: Time granularity ('month' or 'year')

        Returns:
            Dictionary mapping time periods to publication counts
        """
        timeline = defaultdict(int)

        for article in articles_metadata:
            pub_date = article.get("publication_date")

            if pub_date:
                # Parse date
                if isinstance(pub_date, str):
                    try:
                        pub_date = datetime.fromisoformat(pub_date)
                    except:
                        continue

                # Format based on granularity
                if granularity == "year":
                    period = str(pub_date.year)
                else:  # month
                    period = pub_date.strftime("%Y-%m")

                timeline[period] += 1

        # Sort by period
        sorted_timeline = dict(sorted(timeline.items()))

        logger.info(
            f"Created publication timeline with {len(sorted_timeline)} {granularity} periods"
        )
        return sorted_timeline

    def get_author_statistics(
        self, articles_metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze author statistics.

        Args:
            articles_metadata: List of article metadata dictionaries

        Returns:
            Dictionary with author statistics
        """
        author_counts = Counter()
        total_articles = len(articles_metadata)
        author_collaborations = defaultdict(set)

        for article in articles_metadata:
            authors = article.get("authors", [])

            # Count individual authors
            for author in authors:
                author_counts[author] += 1

            # Track collaborations
            if len(authors) > 1:
                for author in authors:
                    # Add other co-authors
                    author_collaborations[author].update(
                        set(authors) - {author}
                    )

        # Get top authors
        top_authors = author_counts.most_common(10)

        # Calculate average authors per paper
        total_authors = sum(len(article.get("authors", [])) for article in articles_metadata)
        avg_authors = total_authors / total_articles if total_articles > 0 else 0

        return {
            "top_authors": top_authors,
            "total_unique_authors": len(author_counts),
            "avg_authors_per_paper": avg_authors,
            "most_collaborative_authors": [
                (author, len(collaborators))
                for author, collaborators in sorted(
                    author_collaborations.items(), key=lambda x: len(x[1]), reverse=True
                )[:10]
            ],
        }

    def get_article_type_distribution(
        self, articles_metadata: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Get distribution of article types.

        Args:
            articles_metadata: List of article metadata dictionaries

        Returns:
            Dictionary mapping article types to counts
        """
        type_counts = Counter()

        for article in articles_metadata:
            article_type = article.get("article_type", "Unknown")
            type_counts[article_type] += 1

        return dict(type_counts)

    def find_related_articles(
        self,
        articles_metadata: List[Dict[str, Any]],
        reference_article: Dict[str, Any],
        top_n: int = 5,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find articles related to a reference article based on keyword overlap.

        Args:
            articles_metadata: List of article metadata dictionaries
            reference_article: Reference article to find related articles for
            top_n: Number of related articles to return

        Returns:
            List of (article, similarity_score) tuples
        """
        ref_keywords = set(
            kw.lower() for kw in reference_article.get("keywords", [])
        )
        ref_doi = reference_article.get("doi")

        if not ref_keywords:
            return []

        similarity_scores = []

        for article in articles_metadata:
            # Skip the reference article itself
            if article.get("doi") == ref_doi:
                continue

            article_keywords = set(kw.lower() for kw in article.get("keywords", []))

            if article_keywords:
                # Calculate Jaccard similarity
                intersection = len(ref_keywords & article_keywords)
                union = len(ref_keywords | article_keywords)
                similarity = intersection / union if union > 0 else 0

                if similarity > 0:
                    similarity_scores.append((article, similarity))

        # Sort by similarity and return top N
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Found {len(similarity_scores)} related articles")
        return similarity_scores[:top_n]
