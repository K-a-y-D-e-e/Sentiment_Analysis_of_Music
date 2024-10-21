# Sentiment_Analysis_of_Music

Workflow

    Data Collection (Web Scraping)
        Responsibility: Sahishnu Raut
        Task: Web scrape lyrics from Genius, focusing on Kendrick Lamar's songs. Use libraries like BeautifulSoup, Scrapy, or Selenium to automate the extraction of lyrics.
            Tools: Python, BeautifulSoup/Scrapy, Requests, Genius API (optional)
            Expected Output: A CSV or JSON file containing song titles, lyrics, and metadata (e.g., album, release year).

    Data Preprocessing
        Responsibility: Ron Mathew Jobi
        Task: Clean the dataset to prepare it for analysis and model training.
            Steps:
                Text Cleaning: Remove punctuation, special characters, and redundant words like stop words (e.g., “the,” “and,” “is”).
                Lowercasing: Convert all text to lowercase to maintain consistency.
                Tokenization: Split lyrics into individual words or phrases.
                Lemmatization/Stemming: Convert words to their root forms to reduce vocabulary size.
                Additional Filtering: Optionally remove non-informative lyrics like repetitive choruses, ad-libs, or profanity.
            Tools: NLTK, SpaCy, Regex.
            Expected Output: A clean, tokenized dataset ready for analysis or model training.

    Model Fine-tuning & Training
        Responsibility: Person 3
        Task: Fine-tune a Large Language Model (LLM) to improve its understanding of lyrics and sentiment analysis.
            Steps:
                Model Selection: Choose an LLM like GPT, BERT, or a music-specific model.
                Fine-tuning: Use your cleaned dataset to fine-tune the LLM. Focus on transfer learning techniques to ensure the model adapts to lyrical content.
                Training: Train the model on a GPU or cloud platform (e.g., Google Colab, AWS).
                Evaluation: Test the model's performance using metrics like accuracy, precision, recall, and F1-score for sentiment analysis tasks.
            Tools: Hugging Face, PyTorch, TensorFlow, Google Colab, Transformers library.
            Expected Output: A fine-tuned LLM capable of analyzing and explaining lyrics in terms of sentiment.
