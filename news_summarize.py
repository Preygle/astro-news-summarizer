from transformers import pipeline
import os, torch
from news_scrape import get_astronomy_articles


def setup_local_qwen_model():
    
    # Ensure the model directory exists and download the Qwen 2.5 7B model if not already present
    model_dir = "local_qwen2.5_7b_model"

    # Check if the model directory exists
    if not os.path.exists(model_dir):

        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("First time setup: Downloading Qwen 2.5 7B model...")

        model_name = "Qwen/Qwen2.5-7B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Save the model and tokenizer to the local directory
        os.makedirs(model_dir, exist_ok=True)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"Qwen 2.5 7B model saved to {model_dir}")
    else:
        print("Qwen 2.5 7B model already exists locally!")


def load_local_qwen_summarizer():

    # Ensure the model is set up
    model_dir = "local_qwen2.5_7b_model"
    device = 0 if torch.cuda.is_available() else -1

    # Load the summarization pipeline with the local model
    summarizer = pipeline(
        "text-generation",  
        model=model_dir,
        tokenizer=model_dir,
        device=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    return summarizer


def summarize_with_qwen(text, summarizer):
    """Custom summarization function for Qwen 2.5 7B"""
    # Create a proper prompt for Qwen
    prompt = f"""<|im_start|>user
Please provide a concise 2-3 sentence summary of this astronomy news article:

{text[:1500]}<|im_end|>
<|im_start|>assistant
"""

    try:
        # Generate response
        output = summarizer(
            prompt,
            max_new_tokens=150,  # Changed from max_length
            do_sample=False,
            temperature=0.1,
            pad_token_id=summarizer.tokenizer.eos_token_id
        )

        # Extract the summary from generated text
        generated_text = output[0]['generated_text']
        summary = generated_text.split("<|im_start|>assistant")[-1].strip()

        return summary

    except Exception as e:
        print(f"Error in summarization: {e}")
        return "Summary generation failed."
    

def summarize_articles():

    # Setup model locally
    setup_local_qwen_model()

    # Load summarizer
    print("Loading Qwen 2.5 7B summarizer...")
    summarizer = load_local_qwen_summarizer()

    # Import articles
    from news_scrape import get_astronomy_articles

    print("Fetching articles...")
    articles = get_astronomy_articles()

    if not articles:
        print("No articles found!")
        return

    print(f"\nSummarizing {len(articles)} articles with Qwen 2.5 7B...")

    for i, article in enumerate(articles, 1):
        try:
            if article['content'] and len(article['content'].strip()) > 50:
                # Use the custom Qwen summarization function
                summary_text = summarize_with_qwen(
                    article['content'], summarizer)

                print(f"\n--- Article {i} ---")
                print(f"Title: {article['title']}")

                # Store the summary in the article dictionary
                articles[i-1]['summary'] = summary_text
                print(f"Summary: {articles[i-1]['summary']}")
                print("-" * 50)


            else:
                print(f"Skipping article {i}: Content too short")

        except Exception as e:
            print(f"Error summarizing article {i}: {e}")
            continue

# Main execution
if __name__ == "__main__":
    summarize_articles()
