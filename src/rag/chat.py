"""
Chat interface for RAG system using Claude.
"""

from typing import List, Dict, Any, Optional

import anthropic
from loguru import logger

from src.rag.retriever import Retriever
from config.settings import settings


class RAGChat:
    """RAG-powered chat interface."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize RAG chat.

        Args:
            retriever: Retriever instance (creates new if None)
            api_key: Anthropic API key (defaults to settings)
            model: Claude model to use (defaults to settings)
        """
        self.retriever = retriever or Retriever()
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or settings.llm_model
        self.client = anthropic.Anthropic(api_key=self.api_key)

        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []

        logger.info(f"Initialized RAGChat with model: {self.model}")

    def chat(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Process a chat query with RAG.

        Args:
            query: User query
            top_k: Number of documents to retrieve
            filters: Optional metadata filters
            temperature: LLM temperature (defaults to settings)
            max_tokens: Maximum tokens in response

        Returns:
            Dictionary with response, sources, and metadata
        """
        logger.info(f"Processing query: {query[:100]}...")

        # Retrieve relevant documents
        documents = self.retriever.retrieve(query, top_k=top_k, filters=filters)

        if not documents:
            logger.warning("No documents retrieved")
            return {
                "response": "I couldn't find any relevant information in the corpus to answer your question.",
                "sources": [],
                "num_sources": 0,
            }

        # Format context
        context = self.retriever.format_context(documents)

        # Build prompt
        system_prompt = self._build_system_prompt()
        user_message = self._build_user_message(query, context)

        # Call Claude
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or settings.max_tokens,
                temperature=temperature or settings.llm_temperature,
                system=system_prompt,
                messages=self.conversation_history + [{"role": "user", "content": user_message}],
            )

            answer = response.content[0].text

            # Extract source information
            sources = self._extract_sources(documents)

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": answer})

            logger.info("Generated response successfully")

            return {
                "response": answer,
                "sources": sources,
                "num_sources": len(documents),
                "retrieved_documents": documents,
            }

        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return {
                "response": f"Error generating response: {str(e)}",
                "sources": [],
                "num_sources": 0,
            }

    def chat_stream(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Process a chat query with streaming response.

        Args:
            query: User query
            top_k: Number of documents to retrieve
            filters: Optional metadata filters
            temperature: LLM temperature
            max_tokens: Maximum tokens in response

        Yields:
            Response chunks as they arrive
        """
        logger.info(f"Processing streaming query: {query[:100]}...")

        # Retrieve relevant documents
        documents = self.retriever.retrieve(query, top_k=top_k, filters=filters)

        if not documents:
            yield "I couldn't find any relevant information in the corpus to answer your question."
            return

        # Format context
        context = self.retriever.format_context(documents)

        # Build prompt
        system_prompt = self._build_system_prompt()
        user_message = self._build_user_message(query, context)

        # Call Claude with streaming
        try:
            full_response = ""

            with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens or settings.max_tokens,
                temperature=temperature or settings.llm_temperature,
                system=system_prompt,
                messages=self.conversation_history + [{"role": "user", "content": user_message}],
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    yield text

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": full_response})

        except Exception as e:
            logger.error(f"Failed to generate streaming response: {e}")
            yield f"Error generating response: {str(e)}"

    def reset_conversation(self) -> None:
        """Reset conversation history."""
        self.conversation_history = []
        logger.info("Reset conversation history")

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get current conversation history."""
        return self.conversation_history.copy()

    def _build_system_prompt(self) -> str:
        """Build system prompt for Claude."""
        return """You are a helpful research assistant analyzing academic articles about climate science and environmental research.

Your role is to:
1. Answer questions based on the provided article excerpts
2. Cite specific sources when making claims
3. Acknowledge uncertainty when information is incomplete
4. Provide clear, accurate, and well-structured responses

Guidelines:
- Always cite sources using the [Source N] format provided
- Distinguish between facts from the articles and your general knowledge
- If the articles don't contain relevant information, say so clearly
- Provide context and explain technical terms when helpful
- Be concise but thorough"""

    def _build_user_message(self, query: str, context: str) -> str:
        """Build user message with query and context."""
        return f"""Based on the following excerpts from academic articles, please answer this question:

Question: {query}

Relevant Article Excerpts:
{context}

Please provide a comprehensive answer citing the relevant sources."""

    def _extract_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract source information from documents."""
        sources = []

        for doc in documents:
            metadata = doc.get("metadata", {})
            source = {
                "title": metadata.get("title", "Unknown"),
                "doi": metadata.get("doi", "Unknown"),
                "authors": metadata.get("authors", "Unknown"),
                "publication_date": metadata.get("publication_date", "Unknown"),
                "url": metadata.get("url", ""),
            }
            sources.append(source)

        return sources

    def ask_with_conversation(
        self, query: str, include_history: bool = True, **kwargs
    ) -> Dict[str, Any]:
        """
        Ask a question, optionally using conversation history.

        Args:
            query: User query
            include_history: Whether to include conversation history
            **kwargs: Additional arguments passed to chat()

        Returns:
            Response dictionary
        """
        if not include_history:
            # Temporarily clear history
            old_history = self.conversation_history
            self.conversation_history = []

            result = self.chat(query, **kwargs)

            # Restore history but don't add this interaction
            self.conversation_history = old_history
            return result
        else:
            return self.chat(query, **kwargs)
