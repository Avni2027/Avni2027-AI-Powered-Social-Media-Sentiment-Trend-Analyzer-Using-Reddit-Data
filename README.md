# Reddit AI/ML Sentiment and Trend Analyzer

## Project Overview
A real-time Social Media Sentiment Analyzer that collects live Reddit posts from AI/ML communities, performs Natural Language Processing (NLP), analyzes sentiment, extracts trending keywords, and displays everything in an interactive web dashboard.

## Live Dashboard Features
- Sentiment Analysis (Positive / Negative / Neutral)
- Interactive Charts and Visualizations
- Trending Keyword Extraction
- News Headlines Integration
- Subreddit Filter and Sentiment Filter
- Post Search Functionality
- WordCloud of Most Used Terms

## Data Collection
- 878 unique Reddit posts collected from 8 subreddits
- 50 AI/ML news headlines from TechCrunch, Wired, The Verge
- No Reddit API key needed — uses Reddit JSON endpoint
- Data collected from r/MachineLearning, r/LocalLLaMA, r/artificial, r/OpenAI, r/technology, r/ChatGPT, r/deeplearning, r/singularity

## NLP Pipeline
- Text Cleaning with NLTK
- Stop word removal with AI/ML term whitelisting
- Sentiment Analysis using VADER
- Keyword Extraction using TF-IDF

## Sentiment Results
- Total Posts  : 878
- Positive     : 254 (28.9%)
- Negative     : 187 (21.3%)
- Neutral      : 437 (49.8%)

## Tech Stack
- Python — Core programming language
- Streamlit — Interactive web dashboard
- NLTK — Text preprocessing
- VADER Sentiment — Sentiment analysis
- scikit-learn — TF-IDF keyword extraction
- Plotly — Interactive charts
- WordCloud — Keyword visualization
- pandas — Data processing
- requests — Reddit data collection
- NewsAPI — News headlines collection

## Visualizations
1. Sentiment Distribution Pie Chart
2. Posts Per Subreddit Bar Chart
3. Sentiment Breakdown Per Subreddit
4. Engagement Scatter Score vs Comments
5. Sentiment Score Distribution Histogram
6. Top 15 Trending Keywords Bar Chart
7. News Headlines Sentiment Pie Chart
8. AI/ML Terms WordCloud

## How to Run

Step 1 - Install Dependencies
pip install streamlit requests pandas plotly vaderSentiment nltk newsapi-python matplotlib wordcloud scikit-learn numpy

Step 2 - Download NLTK Data
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

Step 3 - Run Jupyter Notebook
Run all cells in reddit_sentiment_project.ipynb

Step 4 - Launch Dashboard
streamlit run dashboard.py

Step 5 - Open Browser
http://localhost:8501

## Key Learnings
- Reddit JSON API trick for data collection without API key
- NLP preprocessing pipeline for social media text
- VADER sentiment analysis for short informal text
- TF-IDF for extracting meaningful trending keywords
- Building interactive dashboards with Streamlit and Plotly

## Author
- Name  : Your Name Here
- Email : Your Email Here
- GitHub: Your GitHub Here

## Acknowledgements
- Reddit JSON API for public data access
- NewsAPI for news headlines
- VADER Sentiment Analysis library
- Streamlit for dashboard framework
