"""
Embedding generation using Voyage AI.
"""

from typing import List, Optional

import voyageai
from loguru import logger
from tqdm import tqdm

from config.settings import settings


class EmbeddingGenerator:
    """Generates embeddings for text using Voyage AI."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Initialize embedding generator.

        Args:
            api_key: Voyage AI API key (defaults to settings)
            model: Embedding model to use (defaults to settings)
            batch_size: Batch size for generation (defaults to settings)
        """
        self.api_key = api_key or settings.voyage_api_key
        self.model = model or settings.embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size

        # Initialize Voyage AI client
        self.client = voyageai.Client(api_key=self.api_key)

        logger.info(f"Initialized EmbeddingGenerator with model: {self.model}")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        result = self.client.embed(
            texts=[text], model=self.model, input_type="document"
        )
        return result.embeddings[0]

    def embed_texts(
        self, texts: List[str], show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        logger.info(f"Generating embeddings for {len(texts)} texts")

        all_embeddings = []

        # Process in batches
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating embeddings")

        for i in iterator:
            batch = texts[i : i + self.batch_size]

            try:
                result = self.client.embed(
                    texts=batch, model=self.model, input_type="document"
                )
                all_embeddings.extend(result.embeddings)

            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch {i}: {e}")
                # Add empty embeddings for failed batch
                all_embeddings.extend([[0.0] * 1024] * len(batch))

        logger.info(f"Generated {len(all_embeddings)} embeddings")
        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.

        Args:
            query: Search query text

        Returns:
            Embedding vector
        """
        result = self.client.embed(
            texts=[query], model=self.model, input_type="query"
        )
        return result.embeddings[0]

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings for the current model.

        Returns:
            Embedding dimension
        """
        # Voyage-2 returns 1024-dimensional embeddings
        if "voyage-2" in self.model:
            return 1024
        # voyage-large-2 returns 1536-dimensional embeddings
        elif "voyage-large-2" in self.model:
            return 1536
        else:
            # Default, will be determined from first embedding
            return 1024


def generate_embeddings(
    texts: List[str], show_progress: bool = True
) -> List[List[float]]:
    """
    Convenience function to generate embeddings.

    Args:
        texts: List of texts to embed
        show_progress: Whether to show progress bar

    Returns:
        List of embedding vectors
    """
    generator = EmbeddingGenerator()
    return generator.embed_texts(texts, show_progress=show_progress)
