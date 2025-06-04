from transformers import pipeline, AutoTokenizer
import os
import json
from news_scrape import get_astronomy_articles
from datetime import datetime


def setup_local_model():
    model_dir = "local_falconsai_model"

    if not os.path.exists(model_dir):
        print("First time setup: Downloading model...")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_name = "Falconsai/text_summarization"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        os.makedirs(model_dir, exist_ok=True)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"Model saved to {model_dir}")
    else:
        print("Model already exists locally!")


def load_local_summarizer():
    """Load the local summarization model and tokenizer"""
    model_dir = "local_falconsai_model"
    return pipeline("summarization", model=model_dir, tokenizer=model_dir)


def smart_chunk_text(text, tokenizer, max_tokens=400):
    """Intelligently chunk text by sentences to stay under token limit"""
    sentences = text.split('.')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        test_chunk = current_chunk + ". " + sentence if current_chunk else sentence
        tokens = tokenizer.tokenize(test_chunk)

        if len(tokens) <= max_tokens:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def summarize_single_article(article_content, summarizer=None, tokenizer=None):
    """Summarize a single article - this is what Streamlit should call"""

    if summarizer is None or tokenizer is None:
        setup_local_model()
        tokenizer = AutoTokenizer.from_pretrained("local_falconsai_model")
        summarizer = load_local_summarizer()

    # Check if article is short enough to summarize directly
    tokens = tokenizer.tokenize(article_content)

    if len(tokens) <= 400:  # Safe margin under 512
        try:
            summary = summarizer(
                article_content,
                max_length=120,
                min_length=30,
                do_sample=False,
                truncation=True
            )
            return summary[0]['summary_text']
        except Exception as e:
            print(f"Direct summarization failed: {e}")
            return "Summary generation failed."

    # Article is too long, chunk it
    print(f"Article too long ({len(tokens)} tokens), chunking...")
    chunks = smart_chunk_text(article_content, tokenizer, max_tokens=400)

    if not chunks:
        return "Could not chunk article for summarization."

    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        try:
            print(f"  Summarizing chunk {i+1}/{len(chunks)}")
            summary = summarizer(
                chunk,
                max_length=80,
                min_length=20,
                do_sample=False,
                truncation=True
            )
            chunk_summaries.append(summary[0]['summary_text'])
        except Exception as e:
            print(f"  Chunk {i+1} failed: {e}")
            continue

    if not chunk_summaries:
        return "All chunks failed to summarize."

    # Combine chunk summaries
    combined_summary = " ".join(chunk_summaries)

    # If combined summary is still too long, summarize it again
    combined_tokens = tokenizer.tokenize(combined_summary)
    if len(combined_tokens) > 400:
        try:
            print("  Final summarization of combined chunks...")
            final_summary = summarizer(
                combined_summary,
                max_length=120,
                min_length=40,
                do_sample=False,
                truncation=True
            )
            return final_summary[0]['summary_text']
        except Exception as e:
            print(f"  Final summarization failed: {e}")
            return combined_summary[:500] + "..."

    return combined_summary


def save_articles_to_json(articles, filename="astronomy_summaries_falconsai.json"):
    """Convert articles dictionary to JSON and save to file"""
    json_ready_articles = []

    for article in articles:
        clean_article = {
            'title': article.get('title', ''),
            'url': article.get('url', ''),
            'source': article.get('source', ''),
            'published': str(article.get('published', '')),
            'content': article.get('content', ''),
            'content_length': len(article.get('content', '')),
            'summary': article.get('summary', 'No summary available'),
            'processed_at': datetime.now().isoformat()
        }
        json_ready_articles.append(clean_article)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(json_ready_articles, f, indent=4, ensure_ascii=False)

    print(f"✅ Saved {len(json_ready_articles)} articles to {filename}")
    return json_ready_articles


def summarize_all_articles():
    """Main function to summarize all articles - for command line use"""
    setup_local_model()

    tokenizer = AutoTokenizer.from_pretrained("local_falconsai_model")
    summarizer = load_local_summarizer()

    print("Loading articles...")
    articles = get_astronomy_articles()

    if not articles:
        print("No articles found!")
        return []

    print(f"Found {len(articles)} articles to summarize\n")

    for i, article in enumerate(articles, 1):
        try:
            if not article.get('content') or len(article['content'].strip()) < 50:
                print(f"Skipping article {i}: Content too short")
                continue

            print(f"🔄 Processing Article {i}/{len(articles)}")
            print(f"📰 Title: {article['title']}")

            tokens = tokenizer.tokenize(article['content'])
            print(
                f"📊 Content: {len(article['content'])} chars, {len(tokens)} tokens")

            # Summarize the article
            summary = summarize_single_article(
                article['content'], summarizer, tokenizer)

            # Store and display summary
            article['summary'] = summary
            print(f"✅ Summary: {summary}")
            print("=" * 60)

        except Exception as e:
            print(f"❌ Error processing article {i}: {e}")
            print("=" * 60)
            continue

    if articles:
        save_articles_to_json(articles)

    return articles


if __name__ == "__main__":
    summarize_all_articles()
