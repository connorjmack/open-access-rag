"""
Vector store management using ChromaDB.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from config.settings import settings
from src.processor.text_processor import TextChunk


class VectorStore:
    """Manages vector storage and retrieval using ChromaDB."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize vector store.

        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection to use
        """
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.chroma_collection_name

        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
            ),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Open access journal articles"},
        )

        logger.info(
            f"Initialized VectorStore with collection '{self.collection_name}' at {self.persist_directory}"
        )

    def add_chunks(
        self,
        chunks: List[TextChunk],
        embeddings: List[List[float]],
        article_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add text chunks with embeddings to the vector store.

        Args:
            chunks: List of TextChunk objects
            embeddings: List of embedding vectors
            article_metadata: Optional article-level metadata to merge
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match number of embeddings ({len(embeddings)})"
            )

        if not chunks:
            return

        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        embeddings_list = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create unique ID (DOI + chunk index)
            doi = chunk.metadata.get("doi", "unknown")
            chunk_id = f"{doi}_{chunk.chunk_index}"
            ids.append(chunk_id)

            # Store chunk text
            documents.append(chunk.text)

            # Merge article metadata with chunk metadata
            metadata = {
                **chunk.metadata,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            }

            if article_metadata:
                metadata.update(article_metadata)

            metadatas.append(metadata)
            embeddings_list.append(embedding)

        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings_list,
                metadatas=metadatas,
            )
            logger.info(f"Added {len(chunks)} chunks to vector store")
        except Exception as e:
            logger.error(f"Failed to add chunks to vector store: {e}")
            raise

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the vector store for similar chunks.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Optional metadata filters

        Returns:
            Query results with ids, documents, metadatas, and distances
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
            )

            logger.debug(f"Retrieved {len(results['ids'][0])} results from vector store")
            return results

        except Exception as e:
            logger.error(f"Failed to query vector store: {e}")
            raise

    def get_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific chunk by ID.

        Args:
            chunk_id: Unique chunk identifier

        Returns:
            Chunk data or None if not found
        """
        try:
            result = self.collection.get(ids=[chunk_id])

            if result["ids"]:
                return {
                    "id": result["ids"][0],
                    "document": result["documents"][0],
                    "metadata": result["metadatas"][0],
                }
            return None

        except Exception as e:
            logger.error(f"Failed to get chunk {chunk_id}: {e}")
            return None

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """
        Get metadata for all chunks in the collection.

        Returns:
            List of metadata dictionaries
        """
        try:
            result = self.collection.get()
            return result["metadatas"]

        except Exception as e:
            logger.error(f"Failed to get all metadata: {e}")
            return []

    def count(self) -> int:
        """
        Get the number of chunks in the collection.

        Returns:
            Number of chunks
        """
        return self.collection.count()

    def delete_by_doi(self, doi: str) -> None:
        """
        Delete all chunks for a specific article.

        Args:
            doi: Article DOI
        """
        try:
            # ChromaDB doesn't support direct deletion by metadata
            # We need to get all IDs first and then delete
            results = self.collection.get(where={"doi": doi})

            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks for DOI: {doi}")

        except Exception as e:
            logger.error(f"Failed to delete chunks for DOI {doi}: {e}")

    def clear(self) -> None:
        """Clear all data from the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Open access journal articles"},
            )
            logger.info("Cleared vector store")

        except Exception as e:
            logger.error(f"Failed to clear vector store: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with statistics
        """
        count = self.count()
        metadatas = self.get_all_metadata()

        # Extract unique DOIs
        dois = set(m.get("doi") for m in metadatas if m.get("doi"))

        # Extract unique journals
        journals = set(m.get("journal") for m in metadatas if m.get("journal"))

        # Extract date range
        dates = [
            m.get("publication_date") for m in metadatas if m.get("publication_date")
        ]
        date_range = (min(dates), max(dates)) if dates else (None, None)

        return {
            "total_chunks": count,
            "unique_articles": len(dois),
            "journals": list(journals),
            "date_range": date_range,
        }

    def search_by_text(
        self, query_text: str, n_results: int = 5
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Search using text query (requires embedding generation).

        Args:
            query_text: Query text
            n_results: Number of results to return

        Returns:
            List of (document, metadata, distance) tuples
        """
        # This is a placeholder - actual implementation should use the embedding generator
        # For now, return empty results
        logger.warning(
            "search_by_text requires embedding generation - use query() with pre-generated embeddings instead"
        )
        return []

    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export collection data to a dictionary.

        Returns:
            Dictionary with all collection data
        """
        result = self.collection.get()
        return {
            "ids": result["ids"],
            "documents": result["documents"],
            "metadatas": result["metadatas"],
            "embeddings": result.get("embeddings", []),
        }
