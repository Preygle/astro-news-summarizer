import feedparser
from newspaper import Article
from newspaper import Config
import time
from datetime import datetime

# Configure newspaper3k for better reliability
config = Config()
config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
config.request_timeout = 10

def get_astronomy_articles():
    """
    Fetches latest astronomy articles from multiple RSS feeds
    Returns list of article dictionaries with full content
    """

    # Major astronomy RSS feeds
    rss_feeds = [
        "https://www.nasa.gov/feed/",
        "https://www.nasa.gov/news-release/feed/",
        "https://www.astronomy.com/tags/sky-this-week/feed/",
        "https://www.astronomy.com/tags/news/feed/"
    ]  # "https://www.space.com/feeds/all" # Space.com feed can be added if needed (does generate a lot of advert content)

    articles = []  # List to hold all articles

    for feed_url in rss_feeds: # iterate through each feed
        
        try:
            print(f"Fetching articles from {feed_url}...")  
            feed = feedparser.parse(feed_url) #

            for entry in feed.entries[:2]: #get only the latest 5 articles
                article_url = entry.link

                try:
                    article = Article(article_url, config=Config())
                    article.download()
                    article.parse()
                    
                    # Extracting article data
                    article_data = {
                        'title': entry.title,
                        'url': article_url,
                        'published': entry.published if 'published' in entry else None,
                        'content': article.text,
                        'summary': "",
                        'authors': article.authors if article.authors else [],
                        'source': feed.feed.title
                    }
                    if(article_data['title'] in [a['title'] for a in articles]):
                        print(f"Skipping duplicate article: {article_data['title']}")
                        continue
                    articles.append(article_data)

                    # time.sleep(1) # Respectful delay between requests

                except Exception as e:
                    print(f"Error processing article {article_url}: {e}")
                    continue

        except Exception as e:
            print(f"Error fetching feed {feed_url}: {e}")
            continue
        
    return articles


if __name__ == "__main__":
    articles = get_astronomy_articles()
    print(f"\nTotal articles collected: {len(articles)}")

    print(articles[0])

# Preview first article
    # for article in articles:
    #     print(f"\nSample Article:")
    #     print(f"Title: {article['title']}")
    #     print(f"Summary: {article['summary']}")
    #     print(f"Text Preview: {article['content'][:200]}...")
