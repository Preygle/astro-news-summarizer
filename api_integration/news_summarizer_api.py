import sys, os
import json
import requests
from dotenv import load_dotenv



# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now you can import from parent
from news_scrape import get_astronomy_articles

load_dotenv()


def deepseek_summarizer_requests(article_text, max_tokens=150):
    """Summarize article using DeepSeek via OpenRouter with requests library"""

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found in environment variables")

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json", 
        },
        data=json.dumps({
            # Removed :free for better reliability
            "model": "deepseek/deepseek-r1-0528-qwen3-8b",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert astronomy news summarizer. Provide concise, accurate summaries that capture key scientific findings."
                },
                {
                    "role": "user",
                    "content": f"Please provide a concise 2-3 sentence summary of this astronomy news article:\n\n{article_text[:1500]}"
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "stream": False
        })
    )

    # Handle the response
    if response.status_code == 200:
        result = response.json()
        summary = result["choices"][0]["message"]["content"]
        return summary.strip()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return f"Summarization failed: {response.status_code}"


def test_deepseek_summarizer():
    """Test the DeepSeek summarizer"""
    test_article = """
    NASA's James Webb Space Telescope has made a groundbreaking discovery of water vapor 
    in the atmosphere of exoplanet K2-18 b, located 120 light-years away in the constellation Leo. 
    This sub-Neptune exoplanet orbits within the habitable zone of its cool dwarf star, 
    where liquid water could potentially exist on its surface. The detection was made using 
    Webb's Near-Infrared Spectrograph, which analyzed starlight filtered through the planet's 
    atmosphere as it passed in front of its host star. Scientists also detected traces of 
    methane and carbon dioxide, suggesting a hydrogen-rich atmosphere. This discovery represents 
    a significant step forward in the search for potentially habitable worlds beyond our solar system.
    """

    print("Testing DeepSeek summarizer with requests...")
    summary = deepseek_summarizer_requests(test_article)
    print(f"\nOriginal article length: {len(test_article)} characters")
    print(f"\nSummary: {summary}")

# Integration with your existing news scraper


def summarize_astronomy_news_with_requests():
    """Summarize astronomy news using DeepSeek via requests"""

    # Import your news scraper (adjust path as needed)
    import sys
    import os
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))

    from news_scrape import get_astronomy_articles

    print("Fetching astronomy articles...")
    articles = get_astronomy_articles()

    if not articles:
        print("No articles found!")
        return

    print(f"\nSummarizing {len(articles)} articles with DeepSeek R1...")

    for i, article in enumerate(articles, 1):
        try:
            if article['content'] and len(article['content'].strip()) > 100:
                print(f"\n--- Article {i}/{len(articles)} ---")
                print(f"Title: {article['title'][:80]}...")

                # Summarize with DeepSeek
                summary = deepseek_summarizer_requests(article['content'])

                # Store summary
                article['summary'] = summary

                print(f"Summary: {summary}")
                print("-" * 60)

            else:
                print(f"Skipping article {i}: Content too short")

        except Exception as e:
            print(f"Error processing article {i}: {e}")
            continue

    return articles


if __name__ == "__main__":
    # Test the summarizer
    test_deepseek_summarizer()

    # Uncomment to run full summarization
    # summarize_astronomy_news_with_requests()
