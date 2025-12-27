"""
Streamlit web application for Open Access RAG system.
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.storage.vector_store import VectorStore
from src.rag.chat import RAGChat
from src.analysis.summarizer import CorpusSummarizer
from src.analysis.topic_modeling import TopicAnalyzer
from src.analysis.visualizations import Visualizer
from config.settings import settings
from loguru import logger


# Page configuration
st.set_page_config(
    page_title="Open Access RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def initialize_components():
    """Initialize and cache system components."""
    try:
        vector_store = VectorStore()
        rag_chat = RAGChat()
        summarizer = CorpusSummarizer()
        analyzer = TopicAnalyzer()
        visualizer = Visualizer()

        return vector_store, rag_chat, summarizer, analyzer, visualizer
    except Exception as e:
        st.error(f"Failed to initialize components: {e}")
        return None, None, None, None, None


@st.cache_data
def load_corpus_metadata(_vector_store):
    """Load article metadata from vector store."""
    try:
        metadata = _vector_store.get_all_metadata()
        stats = _vector_store.get_stats()
        return metadata, stats
    except Exception as e:
        st.error(f"Failed to load corpus metadata: {e}")
        return [], {}


def main():
    """Main application function."""

    # Title
    st.title("📚 Open Access RAG System")
    st.markdown("*Analyze and interact with academic journal articles*")

    # Initialize components
    vector_store, rag_chat, summarizer, analyzer, visualizer = initialize_components()

    if vector_store is None:
        st.error("Failed to initialize system. Please check your configuration.")
        return

    # Load corpus data
    metadata, stats = load_corpus_metadata(vector_store)

    # Sidebar
    with st.sidebar:
        st.header("Corpus Information")

        if stats:
            st.metric("Total Articles", stats.get("unique_articles", 0))
            st.metric("Total Chunks", stats.get("total_chunks", 0))

            journals = stats.get("journals", [])
            if journals:
                st.write(f"**Journals:** {', '.join(journals)}")

            date_range = stats.get("date_range", (None, None))
            if date_range[0] and date_range[1]:
                st.write(f"**Date Range:** {date_range[0]} to {date_range[1]}")

        st.divider()

        st.header("Settings")
        retrieval_k = st.slider("Documents to Retrieve", 1, 20, 5)
        temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.7, 0.1)

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Visualizations", "📈 Analysis", "ℹ️ About"])

    # Tab 1: Chat Interface
    with tab1:
        st.header("Chat with Your Corpus")

        # Initialize chat history in session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Show sources if available
                if message.get("sources"):
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(
                                f"**{i}. {source.get('title', 'Unknown')}**"
                            )
                            st.caption(
                                f"DOI: {source.get('doi', 'N/A')} | "
                                f"Date: {source.get('publication_date', 'N/A')}"
                            )

        # Chat input
        if prompt := st.chat_input("Ask a question about the corpus..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = rag_chat.chat(
                        prompt, top_k=retrieval_k, temperature=temperature
                    )

                    response = result.get("response", "No response generated.")
                    sources = result.get("sources", [])

                    st.markdown(response)

                    # Show sources
                    if sources:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(
                                    f"**{i}. {source.get('title', 'Unknown')}**"
                                )
                                st.caption(
                                    f"DOI: {source.get('doi', 'N/A')} | "
                                    f"Date: {source.get('publication_date', 'N/A')}"
                                )

                    # Add assistant message
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response,
                            "sources": sources,
                        }
                    )

        # Clear conversation button
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            rag_chat.reset_conversation()
            st.rerun()

    # Tab 2: Visualizations
    with tab2:
        st.header("Corpus Visualizations")

        if not metadata:
            st.warning("No corpus data available. Please fetch and process articles first.")
        else:
            # Publication timeline
            st.subheader("Publication Timeline")
            timeline = analyzer.get_publication_timeline(metadata)
            if timeline:
                fig = visualizer.plot_publication_timeline(timeline)
                st.plotly_chart(fig, use_container_width=True)

            # Keyword trends
            st.subheader("Top Keywords")
            keyword_counts = summarizer.extract_keywords(metadata, num_keywords=20)
            if keyword_counts:
                fig = visualizer.plot_keyword_trends(keyword_counts)
                st.plotly_chart(fig, use_container_width=True)

            # Article type distribution
            st.subheader("Article Type Distribution")
            type_dist = analyzer.get_article_type_distribution(metadata)
            if type_dist:
                fig = visualizer.plot_article_type_distribution(type_dist)
                st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Analysis
    with tab3:
        st.header("Corpus Analysis")

        if not metadata:
            st.warning("No corpus data available. Please fetch and process articles first.")
        else:
            # Generate summary
            st.subheader("Corpus Summary")
            if st.button("Generate Summary"):
                with st.spinner("Analyzing corpus..."):
                    summary = summarizer.summarize_articles(metadata)
                    st.markdown(summary)

            # Extract topics
            st.subheader("Main Topics")
            if st.button("Extract Topics"):
                with st.spinner("Extracting topics..."):
                    topics = summarizer.extract_topics(metadata, num_topics=10)
                    if topics:
                        for i, topic in enumerate(topics, 1):
                            st.write(f"{i}. {topic}")

            # Author statistics
            st.subheader("Author Statistics")
            author_stats = analyzer.get_author_statistics(metadata)
            if author_stats:
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Total Unique Authors", author_stats["total_unique_authors"])
                    st.metric(
                        "Avg Authors per Paper",
                        f"{author_stats['avg_authors_per_paper']:.1f}",
                    )

                with col2:
                    st.write("**Top Authors:**")
                    for author, count in author_stats["top_authors"][:5]:
                        st.write(f"- {author}: {count} publications")

            # Publication trends
            st.subheader("Publication Trends")
            trends = summarizer.analyze_trends(metadata)
            if trends:
                st.write(f"**Total Articles:** {trends['total_articles']}")

                if trends.get("by_year"):
                    st.write("**Publications by Year:**")
                    for year, count in sorted(trends["by_year"].items()):
                        st.write(f"- {year}: {count}")

    # Tab 4: About
    with tab4:
        st.header("About This System")

        st.markdown("""
        ### Open Access RAG System

        This application uses Retrieval-Augmented Generation (RAG) to analyze and interact with
        academic journal articles from open access sources.

        **Features:**
        - 🔍 Semantic search across article corpus
        - 💬 Interactive chat powered by Claude AI
        - 📊 Visual analytics and trend analysis
        - 📈 Topic modeling and keyword extraction

        **Technology Stack:**
        - **LLM:** Claude 3.5 Haiku (Anthropic)
        - **Embeddings:** Voyage AI
        - **Vector Database:** ChromaDB
        - **UI Framework:** Streamlit

        **How it works:**
        1. Articles are fetched from open access journals
        2. Text is processed and chunked
        3. Embeddings are generated and stored in ChromaDB
        4. Users can chat with the corpus using natural language
        5. Claude retrieves relevant context and generates informed responses

        ---

        For more information, see the [GitHub repository](https://github.com/yourusername/open-access-rag).
        """)

        st.subheader("Current Configuration")
        st.code(f"""
LLM Model: {settings.llm_model}
Embedding Model: {settings.embedding_model}
Vector Store: {settings.chroma_persist_dir}
Collection: {settings.chroma_collection_name}
        """)


if __name__ == "__main__":
    main()
