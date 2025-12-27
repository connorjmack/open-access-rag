# Open Access RAG System

A professional-grade Retrieval-Augmented Generation (RAG) system for analyzing open access academic journals. The system fetches articles, generates embeddings, and provides an interactive chat interface to query the corpus along with comprehensive visualizations and analytics.

## Features

- **Automated Article Fetching**: Scrape and download articles from open access journals (starting with PLOS Climate)
- **Semantic Search**: Vector-based search using ChromaDB for efficient retrieval
- **Intelligent Chat Interface**: Query the corpus using Claude AI with contextual responses
- **Visual Analytics**:
  - Topics over time visualization
  - Keyword trend analysis
  - Publication trend charts
  - Corpus summary statistics
- **Modular Architecture**: Extensible design supporting multiple journal sources
- **Web-Based Dashboard**: Interactive Streamlit interface for exploration and analysis

## Tech Stack

- **LLM**: Claude 3.5 Haiku (via Anthropic API)
- **Embeddings**: Voyage AI
- **Vector Database**: ChromaDB (local-first, persistent storage)
- **UI Framework**: Streamlit
- **Language**: Python 3.10+

## Project Structure

```
open-access-rag/
├── config/              # Configuration management
├── src/
│   ├── scraper/        # Article fetching and parsing
│   ├── processor/      # Text processing and embeddings
│   ├── storage/        # Vector database interface
│   ├── analysis/       # Analytics and visualizations
│   ├── rag/           # RAG pipeline and chat
│   └── ui/            # Streamlit web application
├── scripts/           # CLI utilities
├── data/             # Data storage (gitignored)
│   ├── raw/          # Raw article data
│   ├── processed/    # Processed articles
│   └── vectorstore/  # ChromaDB storage
└── tests/            # Unit tests
```

## Installation

### Prerequisites

- Python 3.10 or higher
- Anthropic API key (for Claude)
- Voyage AI API key (for embeddings)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/open-access-rag.git
cd open-access-rag
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

## Configuration

Create a `.env` file in the project root with the following variables:

```env
# Required API Keys
ANTHROPIC_API_KEY=your_anthropic_api_key_here
VOYAGE_API_KEY=your_voyage_api_key_here

# Optional Configuration
NUM_ISSUES=10                    # Number of recent issues to fetch
CHUNK_SIZE=1024                  # Token size for text chunks
CHUNK_OVERLAP=100                # Overlap between chunks
EMBEDDING_MODEL=voyage-2         # Voyage AI model
LLM_MODEL=claude-3-5-haiku-20241022  # Claude model
```

## Usage

### 1. Fetch Articles

Use the CLI script to download articles from a journal:

```bash
python scripts/fetch_articles.py --journal plos-climate --num-issues 10
```

### 2. Process and Embed

Process the downloaded articles and generate embeddings:

```bash
python scripts/process_corpus.py
```

### 3. Launch the Dashboard

Start the Streamlit web interface:

```bash
streamlit run src/ui/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Dashboard Features

### Overview Tab
- Corpus statistics (number of articles, date range, etc.)
- Summary of main topics and themes
- Recent publications

### Visualizations Tab
- **Topics Over Time**: Track how research topics evolve
- **Keyword Trends**: Identify trending keywords and concepts
- **Publication Trends**: Analyze publication frequency and patterns

### Chat Tab
- Interactive Q&A with the corpus
- Context-aware responses powered by Claude
- Source citations for transparency

## Development

### Project Architecture

The system follows a modular pipeline architecture:

1. **Scraper**: Fetches article metadata and full text from journals
2. **Processor**: Cleans text, chunks documents, and generates embeddings
3. **Storage**: Manages vector database operations (store, retrieve, search)
4. **Analysis**: Generates summaries, extracts topics, creates visualizations
5. **RAG**: Implements retrieval and chat functionality
6. **UI**: Provides user interface for interaction

### Adding a New Journal Source

To support a new journal:

1. Create a new scraper in `src/scraper/` that inherits from `BaseScraper`
2. Implement required methods:
   - `fetch_issue_list()`
   - `fetch_article_metadata()`
   - `fetch_article_fulltext()`
3. Register the scraper in the configuration

## API Keys

### Anthropic API
Get your API key from [Anthropic Console](https://console.anthropic.com/)

### Voyage AI
Sign up at [Voyage AI](https://www.voyageai.com/) for embedding API access

## Cost Estimates

Based on analyzing 10 issues (~100-200 articles):

- **Embeddings** (Voyage AI): ~$0.10-0.30
- **Chat** (Claude 3.5 Haiku): ~$0.05-0.15 per session
- **Total**: < $1 for initial corpus processing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Roadmap

- [ ] Support for additional journals (PubMed, arXiv, etc.)
- [ ] Advanced topic modeling with BERTopic
- [ ] Export functionality (PDF reports, CSV data)
- [ ] Multi-language support
- [ ] Caching and incremental updates
- [ ] User authentication and saved sessions

## Acknowledgments

- Built with [Claude](https://www.anthropic.com/claude) by Anthropic
- Embeddings powered by [Voyage AI](https://www.voyageai.com/)
- Vector storage by [ChromaDB](https://www.trychroma.com/)
- UI framework by [Streamlit](https://streamlit.io/)

## Support

For issues and questions, please open an issue on GitHub or contact the maintainers.
