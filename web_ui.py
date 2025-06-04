import streamlit as st
import json
import os
from datetime import datetime

from news_scrape import get_astronomy_articles
from news_summarize import summarize_single_article, setup_local_model, load_local_summarizer
from transformers import AutoTokenizer

ARTICLES_FILE = "astronomy_articles.json"
SUMMARY_FILE = "astronomy_summaries_falconsai.json"


def load_saved_articles():
    """Load articles from JSON file"""
    if os.path.exists(ARTICLES_FILE):
        with open(ARTICLES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_articles(articles):
    """Save articles to JSON file"""
    with open(ARTICLES_FILE, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)


def load_saved_summaries():
    """Load summaries from JSON file"""
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_summaries(summaries):
    """Save summaries to JSON file"""
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)


def fetch_articles():
    """Fetch articles without summarization"""
    with st.spinner("Fetching articles from RSS feeds..."):
        articles = get_astronomy_articles()

    if not articles:
        st.error("No articles found!")
        return []

    # Add fetched timestamp
    for article in articles:
        article['fetched_at'] = datetime.now().isoformat()

    return articles


def generate_summaries(articles):
    """Generate summaries for existing articles"""

    # Setup model
    setup_local_model()

    # Load tokenizer and summarizer
    tokenizer = AutoTokenizer.from_pretrained("local_falconsai_model")
    summarizer = load_local_summarizer()

    # Process each article with progress bar
    progress_bar = st.progress(0)

    for i, article in enumerate(articles):
        progress_bar.progress((i + 1) / len(articles))

        if article.get('content') and len(article['content'].strip()) > 50:
            try:
                summary = summarize_single_article(
                    article['content'], summarizer, tokenizer)
                article['summary'] = summary
                article['summarized_at'] = datetime.now().isoformat()
            except Exception as e:
                article['summary'] = f"Error: {str(e)}"
                article['summarized_at'] = datetime.now().isoformat()
        else:
            article['summary'] = "Content too short"
            article['summarized_at'] = datetime.now().isoformat()

    progress_bar.empty()
    return articles


def articles_page():
    """Page for fetching and displaying articles"""
    st.header("📰 Fetch Articles")
    st.write("Get the latest astronomy news articles from RSS feeds")

    # Buttons for article operations
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Fetch New Articles", type="primary"):
            articles = fetch_articles()
            if articles:
                st.session_state.articles = articles
                save_articles(articles)
                st.success(f"✅ Fetched {len(articles)} articles!")

    with col2:
        if st.button("📂 Load Saved Articles"):
            articles = load_saved_articles()
            if articles:
                st.session_state.articles = articles
                st.success(f"📂 Loaded {len(articles)} articles!")
            else:
                st.warning("No saved articles found.")

    with col3:
        if st.button("🗑️ Clear Articles"):
            if 'articles' in st.session_state:
                del st.session_state.articles
            st.success("Articles cleared!")

    # Display articles
    if "articles" in st.session_state:
        st.subheader(f"📄 Articles ({len(st.session_state.articles)})")

        for i, article in enumerate(st.session_state.articles, 1):
            with st.expander(f"{i}. {article['title'][:80]}..."):

                # Title
                st.subheader(article['title'])

                # Content preview
                content = article.get('content', 'No content available')
                st.write("**Content Preview:**")
                st.text_area("", content[:500] + "..." if len(content) > 500 else content,
                             height=150, disabled=True, key=f"content_{i}")

                # Details
                st.write(f"**Source:** {article.get('source', 'Unknown')}")
                published = article.get('published', 'Unknown')


                if published != 'Unknown':
                        
                    short_date = published[:-5]
                    st.write(f"**Published:** {short_date}")

                if article.get('url'):
                    st.markdown(f"[🔗 Read Full Article]({article['url']})")
    else:
        st.info("👆 Click 'Fetch New Articles' to get started!")


def summaries_page():
    """Page for generating and displaying summaries"""
    st.header("🤖 Generate Summaries")
    st.write("Create AI-powered summaries from your fetched articles")

    # Check if articles are available
    if "articles" not in st.session_state:
        st.warning(
            "⚠️ No articles found! Please fetch articles first from the 'Articles' page.")
        return

    articles = st.session_state.articles
    st.info(f"📄 Found {len(articles)} articles ready for summarization")

    # Buttons for summary operations
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🤖 Generate Summaries", type="primary"):
            with st.spinner("Generating summaries..."):
                summaries = generate_summaries(articles.copy())
                if summaries:
                    st.session_state.summaries = summaries
                    save_summaries(summaries)
                    st.success(
                        f"✅ Generated summaries for {len(summaries)} articles!")

    with col2:
        if st.button("📂 Load Saved Summaries"):
            summaries = load_saved_summaries()
            if summaries:
                st.session_state.summaries = summaries
                st.success(f"📂 Loaded {len(summaries)} summaries!")
            else:
                st.warning("No saved summaries found.")

    with col3:
        if st.button("🗑️ Clear Summaries"):
            if 'summaries' in st.session_state:
                del st.session_state.summaries
            st.success("Summaries cleared!")

    # Display summaries
    if "summaries" in st.session_state:
        st.subheader(f"📝 Summaries ({len(st.session_state.summaries)})")

        for i, article in enumerate(st.session_state.summaries, 1):
            with st.expander(f"{i}. {article['title'][:80]}..."):

                # Title
                st.subheader(article['title'])

                # Summary
                summary = article.get('summary', 'No summary available')
                if summary.startswith('Error:'):
                    st.error(summary)
                elif summary == 'Content too short':
                    st.warning(summary)
                else:
                    st.write("**Summary:**")
                    st.success(summary)

                # Details
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Source:** {article.get('source', 'Unknown')}")
                    if article.get('url'):
                        st.markdown(f"[🔗 Read Full Article]({article['url']})")

                with col2:
                    st.write(
                        f"**Fetched:** {article.get('fetched_at', 'N/A')[:16]}")
                    st.write(
                        f"**Summarized:** {article.get('summarized_at', 'N/A')[:16]}")
    else:
        st.info("👆 Click 'Generate Summaries' to create summaries from your articles!")


def main():
    st.set_page_config(
        page_title="Astronomy News Summarizer",
        page_icon="🔭",
        layout="wide"
    )

    st.title("🔭 Astronomy News Summarizer")
    st.write("Fetch astronomy news and generate AI-powered summaries")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:", ["📰 Articles", "🤖 Summaries"])

    # Page routing
    if page == "📰 Articles":
        articles_page()
    elif page == "🤖 Summaries":
        summaries_page()


if __name__ == "__main__":
    main()
