# LSTM API Machine Learning Project

This project contains a Streamlit dashboard and associated machine learning utilities for time series prediction. The main entry point is `streamlit_main_app.py`.

## Environment Setup

1. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install --upgrade pip
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   Copy `env_example.sh` to `.env` (or source it) and fill in the appropriate values for API keys and settings:
   ```bash
   cp env_example.sh .env
   # edit .env with your values or export variables manually
   ```
   The application expects keys for DART, NewsAPI and other services. Optional database and GPU settings can also be configured in this file. Set `NEWS_API_KEY` if you want the dashboard to fetch real articles for sentiment-based recommendations.

## Launching the Application

Start the Streamlit dashboard using:
```bash
streamlit run streamlit_main_app.py
```
The dashboard will open in your default browser.

## Optional Modules and Features

- **GPU optimization** (`gpu_memory_optimization.py`, `lstm_gpu_improvements.py`)
  - Provides utilities for managing GPU memory and enabling mixed precision. Activate these features by setting `ENABLE_GPU_ACCELERATION=true` in your environment variables.
- **DART API integration** (`dart_api_improvements.py`)
  - Requires a valid `DART_API_KEY` to access financial statements from the Korean market.
- **Market data integration** (`market_data_integration.py`)
  - Pulls data from various sources (e.g., PyKRX, Alpha Vantage). API keys for these services should be placed in your environment configuration.
- **News sentiment analysis** (`news_sentiment.py`)
  - Optional module that fetches recent articles and scores sentiment. Requires a `NEWS_API_KEY` to enable real news recommendations.

All modules are optional and the main Streamlit app will detect their availability based on your environment settings.

