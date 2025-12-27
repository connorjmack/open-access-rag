"""
Embedding generation using Voyage AI or local sentence-transformers.
"""

from typing import List, Optional

from loguru import logger
from tqdm import tqdm

from config.settings import settings


class EmbeddingGenerator:
    """Generates embeddings for text using Voyage AI or local models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        batch_size: Optional[int] = None,
        use_local: Optional[bool] = None,
    ):
        """
        Initialize embedding generator.

        Args:
            api_key: Voyage AI API key (defaults to settings, optional if use_local=True)
            model: Embedding model to use (defaults to settings)
            batch_size: Batch size for generation (defaults to settings)
            use_local: Force local embeddings. If None, auto-detects based on API key availability
        """
        self.batch_size = batch_size or settings.embedding_batch_size

        # Determine whether to use local or API-based embeddings
        if use_local is None:
            # Auto-detect: use local if no API key is available
            self.use_local = not hasattr(settings, 'voyage_api_key') or not settings.voyage_api_key or settings.voyage_api_key == "your_voyage_api_key_here"
        else:
            self.use_local = use_local

        if self.use_local:
            # Use local sentence-transformers
            from sentence_transformers import SentenceTransformer

            self.model_name = model or "all-MiniLM-L6-v2"  # Fast, good quality model
            logger.info(f"Initializing local embedding model: {self.model_name}")

            self.model = SentenceTransformer(self.model_name)
            self.client = None

            logger.info(f"Initialized EmbeddingGenerator with LOCAL model: {self.model_name}")
        else:
            # Use Voyage AI
            import voyageai

            self.api_key = api_key or settings.voyage_api_key
            self.model_name = model or settings.embedding_model
            self.client = voyageai.Client(api_key=self.api_key)
            self.model = None

            logger.info(f"Initialized EmbeddingGenerator with VOYAGE AI model: {self.model_name}")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if self.use_local:
            embedding = self.model.encode([text], convert_to_numpy=True)[0]
            return embedding.tolist()
        else:
            result = self.client.embed(
                texts=[text], model=self.model_name, input_type="document"
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

        if self.use_local:
            # Local model - can process all at once efficiently
            if show_progress:
                logger.info("Using local model for embeddings...")

            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
                batch_size=self.batch_size
            )

            result = [emb.tolist() for emb in embeddings]
            logger.info(f"Generated {len(result)} embeddings")
            return result
        else:
            # Voyage AI - process in batches
            all_embeddings = []

            iterator = range(0, len(texts), self.batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc="Generating embeddings")

            for i in iterator:
                batch = texts[i : i + self.batch_size]

                try:
                    result = self.client.embed(
                        texts=batch, model=self.model_name, input_type="document"
                    )
                    all_embeddings.extend(result.embeddings)

                except Exception as e:
                    logger.error(f"Failed to generate embeddings for batch {i}: {e}")
                    # Add empty embeddings for failed batch
                    dim = self.get_embedding_dimension()
                    all_embeddings.extend([[0.0] * dim] * len(batch))

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
        if self.use_local:
            # For local models, query and document embeddings are the same
            return self.embed_text(query)
        else:
            result = self.client.embed(
                texts=[query], model=self.model_name, input_type="query"
            )
            return result.embeddings[0]

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings for the current model.

        Returns:
            Embedding dimension
        """
        if self.use_local:
            # Get dimension from local model
            if self.model_name == "all-MiniLM-L6-v2":
                return 384
            elif self.model_name == "all-mpnet-base-v2":
                return 768
            else:
                # Try to get from model directly
                return self.model.get_sentence_embedding_dimension()
        else:
            # Voyage-2 returns 1024-dimensional embeddings
            if "voyage-2" in self.model_name:
                return 1024
            # voyage-large-2 returns 1536-dimensional embeddings
            elif "voyage-large-2" in self.model_name:
                return 1536
            else:
                return 1024


def generate_embeddings(
    texts: List[str], show_progress: bool = True, use_local: bool = True
) -> List[List[float]]:
    """
    Convenience function to generate embeddings.

    Args:
        texts: List of texts to embed
        show_progress: Whether to show progress bar
        use_local: Use local embeddings (default: True for free usage)

    Returns:
        List of embedding vectors
    """
    generator = EmbeddingGenerator(use_local=use_local)
    return generator.embed_texts(texts, show_progress=show_progress)
