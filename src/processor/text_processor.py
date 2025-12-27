"""
Text processing utilities for cleaning and chunking article text.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

import tiktoken
from loguru import logger

from config.settings import settings


@dataclass
class TextChunk:
    """A chunk of text with metadata."""

    text: str
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int
    metadata: dict


class TextProcessor:
    """Handles text cleaning and chunking for articles."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        encoding_name: str = "cl100k_base",
    ):
        """
        Initialize text processor.

        Args:
            chunk_size: Target size for each chunk in tokens
            chunk_overlap: Number of overlapping tokens between chunks
            encoding_name: Tokenizer encoding to use
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove special characters but keep punctuation
        text = re.sub(r"[^\w\s.,;:!?()\-\"\']+", "", text)

        # Fix common issues
        text = text.replace("\n\n", "\n")
        text = text.strip()

        return text

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))

    def chunk_text(
        self, text: str, metadata: Optional[dict] = None
    ) -> List[TextChunk]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of TextChunk objects
        """
        if metadata is None:
            metadata = {}

        # Clean the text first
        text = self.clean_text(text)

        # Encode text to tokens
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)

        logger.debug(
            f"Chunking text with {total_tokens} tokens into chunks of {self.chunk_size}"
        )

        chunks = []
        chunk_index = 0
        start_idx = 0

        while start_idx < total_tokens:
            # Calculate end index for this chunk
            end_idx = min(start_idx + self.chunk_size, total_tokens)

            # Extract chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)

            # Calculate character positions (approximate)
            # This is approximate because tokenization doesn't map 1:1 with characters
            start_char = len(self.encoding.decode(tokens[:start_idx]))
            end_char = len(self.encoding.decode(tokens[:end_idx]))

            # Create chunk
            chunk = TextChunk(
                text=chunk_text,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=end_char,
                token_count=len(chunk_tokens),
                metadata=metadata.copy(),
            )

            chunks.append(chunk)

            # Move to next chunk with overlap
            chunk_index += 1
            start_idx += self.chunk_size - self.chunk_overlap

        logger.debug(f"Created {len(chunks)} chunks")
        return chunks

    def chunk_by_sections(
        self, sections: dict, metadata: Optional[dict] = None
    ) -> List[TextChunk]:
        """
        Chunk text that's already divided into sections.

        Args:
            sections: Dictionary mapping section names to text
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of TextChunk objects
        """
        if metadata is None:
            metadata = {}

        all_chunks = []
        chunk_index = 0

        for section_name, section_text in sections.items():
            # Clean section text
            section_text = self.clean_text(section_text)
            tokens = self.encoding.encode(section_text)

            # If section fits in one chunk, keep it together
            if len(tokens) <= self.chunk_size:
                chunk = TextChunk(
                    text=section_text,
                    chunk_index=chunk_index,
                    start_char=0,
                    end_char=len(section_text),
                    token_count=len(tokens),
                    metadata={**metadata, "section": section_name},
                )
                all_chunks.append(chunk)
                chunk_index += 1
            else:
                # Section is too large, split it
                section_chunks = self.chunk_text(
                    section_text, metadata={**metadata, "section": section_name}
                )

                # Update chunk indices
                for chunk in section_chunks:
                    chunk.chunk_index = chunk_index
                    chunk_index += 1

                all_chunks.extend(section_chunks)

        return all_chunks

    def extract_sections(self, text: str) -> dict:
        """
        Attempt to extract sections from article text.

        This is a simple heuristic-based approach.

        Args:
            text: Full article text

        Returns:
            Dictionary mapping section names to text
        """
        sections = {}

        # Common section headers in scientific articles
        section_patterns = [
            r"Abstract",
            r"Introduction",
            r"Methods?",
            r"Materials? and Methods?",
            r"Results?",
            r"Discussion",
            r"Conclusion",
            r"References?",
            r"Acknowledgments?",
        ]

        # Build regex pattern
        pattern = r"^(" + "|".join(section_patterns) + r")\s*$"

        # Split text into lines
        lines = text.split("\n")

        current_section = "Introduction"
        current_text = []

        for line in lines:
            line_stripped = line.strip()

            # Check if line is a section header
            if re.match(pattern, line_stripped, re.IGNORECASE):
                # Save previous section
                if current_text:
                    sections[current_section] = "\n".join(current_text)

                # Start new section
                current_section = line_stripped
                current_text = []
            else:
                current_text.append(line)

        # Save final section
        if current_text:
            sections[current_section] = "\n".join(current_text)

        # If no sections found, return entire text as one section
        if not sections:
            sections["Full Text"] = text

        return sections


def process_article_text(
    fulltext: str,
    metadata: dict,
    use_sections: bool = True,
) -> List[TextChunk]:
    """
    Process article text into chunks.

    Args:
        fulltext: Full article text
        metadata: Article metadata
        use_sections: Whether to try extracting sections first

    Returns:
        List of TextChunk objects
    """
    processor = TextProcessor()

    if use_sections:
        sections = processor.extract_sections(fulltext)
        if len(sections) > 1:
            # Successfully extracted sections
            return processor.chunk_by_sections(sections, metadata)

    # Fall back to simple chunking
    return processor.chunk_text(fulltext, metadata)
