# Open Access RAG System

A RAG (Retrieval-Augmented Generation) system for exploring open access academic journals. Scrapes articles, builds a searchable vector database, and lets you chat with the corpus - all running locally and completely free.

## What it does

- Fetches articles from open access journals (currently supports PLOS Climate)
- Generates embeddings and stores them in a local vector database
- Provides a web interface where you can ask questions about the research
- Shows visualizations like topic trends over time and keyword analysis
- Actually cites its sources instead of making stuff up

## Tech Stack

Everything runs locally on your machine:

- **LLM**: Ollama (llama3.1, mistral, or whatever model you prefer)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Database**: ChromaDB
- **Web UI**: Streamlit
- **Language**: Python 3.10+

No API keys required. No usage costs. Just install and run.

## Project Structure

```
open-access-rag/
├── config/              # Configuration
├── src/
│   ├── scraper/        # Article fetching
│   ├── processor/      # Text processing and embeddings
│   ├── storage/        # Vector database interface
│   ├── analysis/       # Analytics and visualizations
│   ├── rag/           # RAG pipeline and chat
│   └── ui/            # Streamlit app
├── scripts/           # CLI utilities
├── data/             # Data storage (gitignored)
│   ├── raw/          # Raw article data
│   ├── processed/    # Processed articles
│   └── vectorstore/  # ChromaDB storage
└── tests/            # Tests
```

## Installation

### What you need

- Python 3.10 or higher
- Ollama installed and running ([get it here](https://ollama.ai))

### Setup

1. Clone the repo:
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

4. Pull an Ollama model (if you haven't already):
```bash
ollama pull llama3.1
```

That's it. No API keys needed.

### Optional: Using paid APIs instead

If you want to use Claude and Voyage AI instead of the free local models, you can create a `.env` file:

```env
ANTHROPIC_API_KEY=your_key_here
VOYAGE_API_KEY=your_key_here
```

The system will automatically detect the keys and use the paid services. But honestly, the free local setup works fine for most use cases.

## Usage

### 1. Fetch some articles

```bash
python scripts/fetch_articles.py --journal plos-climate --num-issues 10
```

This grabs the most recent 10 issues from PLOS Climate. Takes a few minutes depending on your connection.

### 2. Process and embed them

```bash
python scripts/process_corpus.py
```

This chunks the articles, generates embeddings, and stores everything in ChromaDB. The first run downloads the embedding model (about 80MB), then subsequent runs are quick.

### 3. Start the web interface

```bash
streamlit run src/ui/app.py
```

Opens at `http://localhost:8501`

## What you can do in the dashboard

**Overview tab** - Shows stats about your corpus, main topics, recent publications

**Visualizations tab** - Charts showing how topics and keywords change over time, publication trends

**Chat tab** - Ask questions about the research. It'll retrieve relevant passages and generate answers with citations. Try things like:
- "What are the main challenges in climate modeling?"
- "Summarize recent findings on ocean acidification"
- "What methods are used to measure carbon sequestration?"

## Configuration

You can tweak settings by creating a `.env` file. Here are some useful options:

```env
# Ollama Configuration
OLLAMA_MODEL=llama3.1  # or mistral, phi3, dolphin-mixtral, etc.

# Processing
CHUNK_SIZE=1024
CHUNK_OVERLAP=100

# How many articles to retrieve for each query
RETRIEVAL_TOP_K=5
```

See `.env.example` for all available settings.

## Adding more journal sources

Right now it only supports PLOS Climate, but it's built to be extensible. To add a new journal:

1. Create a new scraper class in `src/scraper/` that inherits from `BaseScraper`
2. Implement the required methods:
   - `fetch_issue_list()` - Get list of issues
   - `fetch_article_metadata()` - Get article info
   - `fetch_article_fulltext()` - Get full article text
3. Register it in the config

The scraper handles different HTML structures, paywalls (or lack thereof), and weird formatting quirks that each journal has.

## Performance

On a typical laptop:
- Fetching 10 issues (100-200 articles): 5-10 minutes
- Processing and embedding: 2-5 minutes
- Chat queries: 2-5 seconds per response

The local embedding model is fast. Ollama response time depends on your hardware - decent on M1/M2 Macs, slower on older CPUs, fast on machines with GPUs.

## Why local/free?

Most RAG tutorials show you how to rack up API costs. This runs entirely on your machine:
- No API rate limits
- No usage costs
- No data leaving your computer
- No internet required after initial setup

The quality is good enough for research exploration, summarization, and learning how RAG systems work.

## License

MIT License - do whatever you want with it.

## Contributing

Pull requests welcome. If you add support for another journal, definitely submit it.

## What's next

Some things that would be cool to add:
- More journals (PubMed, arXiv, bioRxiv, etc.)
- Better topic modeling
- Export functionality
- Incremental updates instead of reprocessing everything
- Multi-language support

## Built with

- [Ollama](https://ollama.ai) - Local LLM runtime
- [sentence-transformers](https://www.sbert.net/) - Embedding models
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Streamlit](https://streamlit.io/) - Web framework

## Questions?

Open an issue if something's broken or confusing.
