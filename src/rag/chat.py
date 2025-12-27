"""
Chat interface for RAG system using Claude or Ollama.
"""

from typing import List, Dict, Any, Optional

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
        use_ollama: Optional[bool] = None,
    ):
        """
        Initialize RAG chat.

        Args:
            retriever: Retriever instance (creates new if None)
            api_key: Anthropic API key (optional if using Ollama)
            model: Model to use (defaults to settings)
            use_ollama: Force Ollama usage. If None, auto-detects based on API key
        """
        self.retriever = retriever or Retriever()

        # Determine whether to use Ollama or Anthropic
        if use_ollama is None:
            # Auto-detect: use Ollama if no API key is available
            self.use_ollama = not hasattr(settings, 'anthropic_api_key') or not settings.anthropic_api_key
        else:
            self.use_ollama = use_ollama

        if self.use_ollama:
            # Use Ollama (free, local)
            import ollama

            self.model = model or getattr(settings, 'ollama_model', 'llama3.1')
            self.client = ollama
            self.api_key = None

            logger.info(f"Initialized RAGChat with OLLAMA model: {self.model}")
        else:
            # Use Anthropic Claude
            import anthropic

            self.api_key = api_key or settings.anthropic_api_key
            self.model = model or settings.llm_model
            self.client = anthropic.Anthropic(api_key=self.api_key)

            logger.info(f"Initialized RAGChat with CLAUDE model: {self.model}")

        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []

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

        # Call LLM
        try:
            if self.use_ollama:
                answer = self._call_ollama(
                    system_prompt, user_message, temperature, max_tokens
                )
            else:
                answer = self._call_anthropic(
                    system_prompt, user_message, temperature, max_tokens
                )

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

    def _call_ollama(
        self,
        system_prompt: str,
        user_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call Ollama API."""
        messages = self.conversation_history + [{"role": "user", "content": user_message}]

        # Add system message at the beginning
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        response = self.client.chat(
            model=self.model,
            messages=full_messages,
            options=options if options else None,
        )

        return response['message']['content']

    def _call_anthropic(
        self,
        system_prompt: str,
        user_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call Anthropic Claude API."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or settings.max_tokens,
            temperature=temperature or settings.llm_temperature,
            system=system_prompt,
            messages=self.conversation_history + [{"role": "user", "content": user_message}],
        )

        return response.content[0].text

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

        # Call LLM with streaming
        try:
            full_response = ""

            if self.use_ollama:
                # Ollama streaming
                messages = self.conversation_history + [{"role": "user", "content": user_message}]
                full_messages = [{"role": "system", "content": system_prompt}] + messages

                options = {}
                if temperature is not None:
                    options["temperature"] = temperature

                stream = self.client.chat(
                    model=self.model,
                    messages=full_messages,
                    stream=True,
                    options=options if options else None,
                )

                for chunk in stream:
                    text = chunk['message']['content']
                    full_response += text
                    yield text

            else:
                # Anthropic streaming
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
        """Build system prompt for the LLM."""
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
