# Cryptocurrency Price Predictor

This application uses machine learning to predict cryptocurrency prices based on historical data from CoinGecko API.

## Features

- Select from multiple cryptocurrencies (Bitcoin, Ethereum, Litecoin, Ripple, Cardano)
- Choose custom date ranges for historical data analysis
- Use different machine learning models (Linear Regression, Random Forest)
- Configure features used for prediction (Moving Averages, Volume, RSI)
- Visualize predictions vs. actual prices
- View detailed model performance metrics

## How It Works

1. Historical cryptocurrency data is fetched from CoinGecko API
2. Features are engineered from price and volume data
3. Machine learning models are trained on historical data
4. Models predict the next day's closing price

## Deployment

This application is configured for deployment on Render. To deploy:

1. Push this repository to GitHub
2. Create a new Web Service in Render
3. Connect to your GitHub repository
4. Render will automatically detect the configuration

## Local Development

To run this application locally:

```
streamlit run app.py
```

## Requirements

- Python 3.11+
- Required packages are listed in requirements.txt