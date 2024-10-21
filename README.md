# Sentiment Analysis of Kendrick Lamar's Music Lyrics using ALBERT

## Project Overview
This project aims to conduct sentiment analysis on Kendrick Lamar's music lyrics by fine-tuning **ALBERT** (A Lite BERT for Self-supervised Learning of Language Representations). ALBERT is a lightweight version of BERT that offers high performance while being computationally efficient. We create a custom dataset by scraping lyrics from Genius, clean and preprocess it, and fine-tune ALBERT for sentiment analysis.

Our objective is to enable deeper insights into socially and politically charged lyrics, particularly within Kendrick Lamar's music.

## Workflow

### 1. Data Collection (Web Scraping)
We scrape lyrics from Genius, focusing on Kendrick Lamar's songs. This includes metadata such as song title, album, and release year. The dataset is stored in CSV or JSON format for further processing.

**Tools Used**: 
- Python
- BeautifulSoup, Scrapy (or other web scraping tools)
- Genius API (optional)

### 2. Data Preprocessing
After collecting the data, we preprocess the lyrics by:
- Removing special characters and redundant words.
- Tokenization and lemmatization.
- Handling repetitive sections like choruses or ad-libs.

**Tools Used**:
- NLTK
- SpaCy
- Regex

### 3. Fine-Tuning ALBERT
We fine-tune **ALBERT** on the preprocessed dataset. ALBERT's lightweight architecture makes it efficient for this task while retaining the accuracy needed for sentiment analysis. 

**Tools Used**:
- Hugging Face Transformers (ALBERT)
- PyTorch or TensorFlow
- Google Colab or AWS for training

### 4. Sentiment Analysis
The fine-tuned ALBERT model will be used to perform sentiment analysis on the lyrics. The model will classify sentiments based on the emotional and thematic content of the lyrics.

## Dataset
Our dataset consists of scraped lyrics from Kendrick Lamar's songs with the following fields:
- **Song Title**
- **Album**
- **Release Year**
- **Lyrics**

The lyrics are preprocessed to remove noise and tokenized for input into the ALBERT model.

## Project Structure
```plaintext
├── data/                   # Directory for storing the scraped lyrics dataset
├── models/                 # Directory for storing fine-tuned ALBERT models
├── notebooks/              # Jupyter notebooks for experiments and EDA
├── scripts/
│   ├── scrape_lyrics.py    # Web scraping script
│   ├── preprocess_data.py  # Data preprocessing script
│   ├── fine_tune_albert.py # ALBERT fine-tuning script
│   ├── evaluate_model.py   # Script for evaluating model performance
└── README.md               # Project documentation

