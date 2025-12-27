"""
Configuration management for the Open Access RAG system.
Uses pydantic-settings for type-safe configuration with validation.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys (Required)
    anthropic_api_key: str = Field(..., description="Anthropic API key for Claude")
    voyage_api_key: str = Field(..., description="Voyage AI API key for embeddings")

    # Journal Configuration
    num_issues: int = Field(default=10, description="Number of recent issues to fetch")
    default_journal: str = Field(
        default="plos-climate", description="Default journal to analyze"
    )

    # Text Processing Configuration
    chunk_size: int = Field(
        default=1024, description="Target token size for text chunks"
    )
    chunk_overlap: int = Field(
        default=100, description="Overlap between consecutive chunks"
    )
    max_chunk_size: int = Field(
        default=2048, description="Maximum allowed chunk size"
    )

    # Embedding Configuration
    embedding_model: str = Field(
        default="voyage-2", description="Voyage AI embedding model"
    )
    embedding_batch_size: int = Field(
        default=128, description="Batch size for embedding generation"
    )

    # LLM Configuration
    llm_model: str = Field(
        default="claude-3-5-haiku-20241022", description="Claude model for chat"
    )
    llm_temperature: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Temperature for LLM generation"
    )
    max_tokens: int = Field(
        default=4096, description="Maximum tokens in LLM response"
    )

    # Vector Store Configuration
    chroma_persist_dir: str = Field(
        default="./data/vectorstore", description="ChromaDB persistence directory"
    )
    chroma_collection_name: str = Field(
        default="articles", description="ChromaDB collection name"
    )

    # RAG Configuration
    retrieval_top_k: int = Field(
        default=5, description="Number of chunks to retrieve for RAG"
    )
    context_window: int = Field(
        default=8000, description="Maximum context window size in tokens"
    )

    # Scraper Configuration
    request_timeout: int = Field(
        default=30, description="HTTP request timeout in seconds"
    )
    rate_limit_delay: float = Field(
        default=1.0, description="Delay between requests in seconds"
    )
    max_retries: int = Field(
        default=3, description="Maximum number of retry attempts"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(
        default="./logs/app.log", description="Log file path"
    )

    # UI Configuration
    streamlit_port: int = Field(
        default=8501, description="Streamlit server port"
    )
    streamlit_theme: str = Field(
        default="light", description="Streamlit theme (light/dark)"
    )

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info) -> int:
        """Ensure chunk overlap is less than chunk size."""
        chunk_size = info.data.get("chunk_size", 1024)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v_upper

    @property
    def data_dir(self) -> Path:
        """Get the data directory path."""
        return Path("./data")

    @property
    def raw_data_dir(self) -> Path:
        """Get the raw data directory path."""
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Get the processed data directory path."""
        return self.data_dir / "processed"

    @property
    def vectorstore_dir(self) -> Path:
        """Get the vector store directory path."""
        return Path(self.chroma_persist_dir)

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)

        if self.log_file:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
