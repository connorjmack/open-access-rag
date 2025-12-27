"""
Retrieval component for RAG system.
"""

from typing import List, Dict, Any, Optional

from loguru import logger

from src.storage.vector_store import VectorStore
from src.processor.embeddings import EmbeddingGenerator
from config.settings import settings


class Retriever:
    """Handles document retrieval for RAG."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        top_k: Optional[int] = None,
    ):
        """
        Initialize retriever.

        Args:
            vector_store: VectorStore instance (creates new if None)
            embedding_generator: EmbeddingGenerator instance (creates new if None)
            top_k: Number of documents to retrieve (defaults to settings)
        """
        self.vector_store = vector_store or VectorStore()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.top_k = top_k or settings.retrieval_top_k

        logger.info(f"Initialized Retriever with top_k={self.top_k}")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve (overrides default)
            filters: Optional metadata filters

        Returns:
            List of retrieved documents with metadata
        """
        k = top_k or self.top_k

        logger.info(f"Retrieving top {k} documents for query: {query[:100]}...")

        # Generate query embedding
        query_embedding = self.embedding_generator.embed_query(query)

        # Search vector store
        results = self.vector_store.query(
            query_embedding=query_embedding, n_results=k, where=filters
        )

        # Format results
        documents = []
        for i in range(len(results["ids"][0])):
            doc = {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
            documents.append(doc)

        logger.info(f"Retrieved {len(documents)} documents")
        return documents

    def retrieve_with_reranking(
        self,
        query: str,
        initial_k: int = 20,
        final_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with optional reranking.

        Currently just retrieves without reranking.
        Can be extended with a reranking model.

        Args:
            query: Search query
            initial_k: Number of documents to initially retrieve
            final_k: Number of documents to return after reranking
            filters: Optional metadata filters

        Returns:
            List of retrieved and reranked documents
        """
        # For now, just do standard retrieval
        # TODO: Add reranking model (e.g., using Cohere or cross-encoder)
        final_k = final_k or self.top_k
        return self.retrieve(query, top_k=min(initial_k, final_k), filters=filters)

    def retrieve_by_metadata(
        self, filters: Dict[str, Any], limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents by metadata filters only (no semantic search).

        Args:
            filters: Metadata filters
            limit: Maximum number of documents to return

        Returns:
            List of matching documents
        """
        logger.info(f"Retrieving documents with filters: {filters}")

        # This requires ChromaDB get() with where clause
        # For now, we'll do a broad query and filter
        # In a real implementation, use direct metadata querying

        all_metadata = self.vector_store.get_all_metadata()

        matching_docs = []
        for i, metadata in enumerate(all_metadata):
            if self._matches_filters(metadata, filters):
                matching_docs.append(
                    {"id": f"doc_{i}", "metadata": metadata, "text": ""}
                )

                if len(matching_docs) >= limit:
                    break

        logger.info(f"Found {len(matching_docs)} matching documents")
        return matching_docs

    def _matches_filters(
        self, metadata: Dict[str, Any], filters: Dict[str, Any]
    ) -> bool:
        """Check if metadata matches filters."""
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def format_context(
        self, documents: List[Dict[str, Any]], max_tokens: Optional[int] = None
    ) -> str:
        """
        Format retrieved documents into context string.

        Args:
            documents: List of retrieved documents
            max_tokens: Maximum tokens to include (defaults to settings)

        Returns:
            Formatted context string
        """
        max_tokens = max_tokens or settings.context_window

        context_parts = []
        current_tokens = 0

        for i, doc in enumerate(documents, 1):
            text = doc["text"]
            metadata = doc.get("metadata", {})

            # Create source citation
            title = metadata.get("title", "Unknown")
            doi = metadata.get("doi", "Unknown")
            source = f"[Source {i}] {title} (DOI: {doi})"

            # Format document
            doc_text = f"{source}\n{text}\n"

            # Estimate tokens (rough approximation: 4 chars = 1 token)
            doc_tokens = len(doc_text) // 4

            if current_tokens + doc_tokens > max_tokens:
                logger.warning(
                    f"Context truncated at {i-1} documents (token limit reached)"
                )
                break

            context_parts.append(doc_text)
            current_tokens += doc_tokens

        context = "\n---\n".join(context_parts)
        logger.debug(f"Formatted context with ~{current_tokens} tokens")

        return context

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the retrieval corpus.

        Returns:
            Dictionary with statistics
        """
        return self.vector_store.get_stats()
