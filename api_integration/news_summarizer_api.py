import sys
import os
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your news scraper
from news_scrape import get_astronomy_articles

import json
import requests
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()


def llama33_summarizer(article_text, max_tokens=200, retries=3):
    """Summarize article using Llama 3.3 8B Instruct Free model"""

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found in environment variables")

    for attempt in range(retries):
        try:
            print(
                f"🔄 Attempt {attempt + 1}/{retries} - Using Llama 3.3 8B Free")

            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": "meta-llama/llama-3.3-8b-instruct:free",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert astronomy and space science summarizer. Provide clear, concise summaries that highlight key scientific discoveries and their significance."
                        },
                        {
                            "role": "user",
                            "content": f"Please summarize this astronomy news article in 2-3 clear sentences, focusing on the main scientific findings:\n\n{article_text[:2000]}"
                        }
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "stream": False
                }),
                timeout=30
            )

            print(f"📡 Response Status: {response.status_code}")

            if response.status_code == 200:
                try:
                    result = response.json()

                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"]

                        if content and content.strip():
                            print(f"✅ Summary generated successfully")
                            return content.strip()
                        else:
                            print("⚠️ Empty content received")
                    else:
                        print("⚠️ No choices in response")

                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error: {e}")

            elif response.status_code == 429:
                print("⏳ Rate limited - waiting 10 seconds...")
                time.sleep(10)
                continue

            elif response.status_code == 503:
                print("⏳ Server busy - waiting 5 seconds...")
                time.sleep(5)
                continue

            else:
                print(f"❌ HTTP Error {response.status_code}: {response.text}")

        except requests.exceptions.Timeout:
            print("⏳ Request timeout - retrying...")

        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {e}")

        # Wait before retry
        if attempt < retries - 1:
            print(f"⏳ Waiting 2 seconds before retry...")
            time.sleep(2)

    return "Summarization failed after multiple attempts"





def summarize_astronomy_news():

    print("🚀 Starting Astronomy News Summarization with Llama 3.3 8B")
    print("="*70)

    # Fetch articles
    print("\n📡 Fetching astronomy articles...")
    articles = get_astronomy_articles()

    if not articles:
        print("❌ No articles found!")
        return []

    print(f"📰 Found {len(articles)} articles to summarize")
    print("-" * 70)

    summarized_articles = []

    for i, article in enumerate(articles, 1):
        try:
            if article.get('content') and len(article['content'].strip()) > 100:
                print(f"\n🔄 Processing Article {i}/{len(articles)}")
                print(f"📰 Title: {article['title'][:70]}...")
                print(
                    f"📊 Content length: {len(article['content'])} characters")

                # Summarize with Llama 3.3 8B
                summary = llama33_summarizer(article['content'])

                if summary and not summary.startswith("❌"):
                    # Store summary in article
                    article['summary'] = summary
                    summarized_articles.append(article)

                    print(f"✅ Summary: {summary}")
                else:
                    print(f"❌ Failed to summarize article {i}")

                print("-" * 50)

                # Rate limiting for free tier
                time.sleep(1)

            else:
                print(f"⏭️ Skipping article {i}: Content too short")

        except Exception as e:
            print(f"❌ Error processing article {i}: {e}")
            continue

    print(
        f"\n🎉 Successfully summarized {len(summarized_articles)}/{len(articles)} articles!")
    return summarized_articles


def save_summaries(articles, filename="llama33_summaries.json"):
    """Save summarized articles to JSON file"""
    try:
        # Prepare data for JSON serialization
        save_data = []
        for article in articles:
            if 'summary' in article:
                save_data.append({
                    'title': article['title'],
                    'summary': article['summary'],
                    'url': article.get('url', ''),
                    'source': article.get('source', ''),
                    'published': str(article.get('published', '')),
                    'content_length': len(article.get('content', ''))
                })

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved {len(save_data)} summaries to {filename}")

        # Also create a readable text file
        text_filename = filename.replace('.json', '.txt')
        with open(text_filename, 'w', encoding='utf-8') as f:
            f.write("ASTRONOMY NEWS SUMMARIES\n")
            f.write("="*50 + "\n\n")

            for i, article in enumerate(save_data, 1):
                f.write(f"{i}. {article['title']}\n")
                f.write(f"   Source: {article['source']}\n")
                f.write(f"   Summary: {article['summary']}\n")
                f.write(f"   URL: {article['url']}\n")
                f.write("-" * 50 + "\n\n")

        print(f".txt file saved to {text_filename}")

    except Exception as e:
        print(f"❌ Error saving summaries: {e}")


if __name__ == "__main__":
    print("🌟 Astronomy News Summarizer with Llama 3.3 8B Free")
    print("="*70)

    # Run the summarizer
    summarized_articles = summarize_astronomy_news()

    if summarized_articles:
        # Save results
        save_summaries(summarized_articles)

        # Display summary statistics
        print(f"\n📊 FINAL STATISTICS:")
        print("="*40)
        print(f"Total articles processed: {len(summarized_articles)}")

        print("\n📋 Quick Preview:")
        for i, article in enumerate(summarized_articles[:3], 1):
            if 'summary' in article:
                print(f"{i}. {article['title'][:50]}...")
                print(f"   {article['summary'][:80]}...")
                print()
    else:
        print("❌ No articles were successfully summarized.")
