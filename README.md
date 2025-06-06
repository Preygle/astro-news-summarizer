# Astronomy News Summarizer 🔭

A comprehensive tool for fetching, processing, and summarizing astronomy news articles using AI models.

## Features ✨

- **Web Interface**: Streamlit-based UI for easy interaction
- **Multiple Summarization Methods**:
  - Local FalconSAI model (default)
  - Llama 3.3 8B via OpenRouter API
- **Article Sources**:
  - NASA feeds
  - Astronomy.com
- **Data Management**:
  - Save/Load articles and summaries
  - JSON export functionality

## Installation 🛠️

1. Clone the repository:
```bash
git clone https://github.com/yourusername/astro-news-summarizer.git
cd astro-news-summarizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r api_integration/requirements.txt
```

3. Set up environment variables:
```bash
cp api_integration/.env.example api_integration/.env
```
Edit the `.env` file with your API keys.

## Usage 🚀

### Web Interface
```bash
streamlit run web_ui.py
```

### Command Line
```bash
python news_summarize.py
```

### API Integration
```bash
python api_integration/news_summarizer_api.py
```

## Configuration ⚙️

### Environment Variables
- `OPENROUTER_API_KEY`: Required for Llama 3.3 API summarization

### Local Model Setup
The first run will automatically download and cache the FalconSAI model (~1.5GB).

## File Structure 📁

```
├── api_integration/            # API integration components
│   ├── .env                    # API configuration
│   ├── news_summarizer_api.py  # API summarization logic
│   └── requirements.txt        # API dependencies
├── local_falconsai_model/      # Local AI model storage
├── web_ui.py                   # Streamlit interface
├── news_summarize.py           # Core summarization logic
├── news_scrape.py              # Article fetching
├── requirements.txt            # Main dependencies
└── README.md                   # This file
```

## Dependencies 📦

### Core Requirements
- streamlit
- newspaper3k
- beautifulsoup4
- transformers
- torch
- nltk

### API Requirements
- openai
- python-dotenv
- requests

## Contributing 🤝

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## License 📄

MIT License
