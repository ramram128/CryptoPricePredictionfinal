import pandas as pd
import requests
from datetime import datetime, timedelta
import time

def fetch_crypto_data(crypto_id, start_date, end_date):
    """
    Fetch historical cryptocurrency data from CoinGecko API.
    
    Args:
        crypto_id (str): The ID of the cryptocurrency (e.g., "bitcoin")
        start_date (datetime): Start date for historical data
        end_date (datetime): End date for historical data
        
    Returns:
        pandas.DataFrame: Dataframe with historical price data
    """
    try:
        # Convert dates to UNIX timestamps
        start_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        # Add one day to end_date to include the end date in the results
        end_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())
        
        # CoinGecko API URL for historical market data
        url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart/range"
        
        # Parameters for the API request
        params = {
            "vs_currency": "usd",
            "from": start_timestamp,
            "to": end_timestamp
        }
        
        # Make the API request
        response = requests.get(url, params=params)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            
            # Extract prices, market caps, and volumes
            prices = data.get("prices", [])
            market_caps = data.get("market_caps", [])
            total_volumes = data.get("total_volumes", [])
            
            # Create dataframes for each data type
            df_prices = pd.DataFrame(prices, columns=["timestamp", "price"])
            df_market_caps = pd.DataFrame(market_caps, columns=["timestamp", "market_cap"])
            df_volumes = pd.DataFrame(total_volumes, columns=["timestamp", "volume"])
            
            # Merge the dataframes on timestamp
            df = pd.merge(df_prices, df_market_caps, on="timestamp")
            df = pd.merge(df, df_volumes, on="timestamp")
            
            # Convert timestamp to datetime
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
            
            # Drop timestamp column and set date as index
            df = df.drop("timestamp", axis=1)
            
            # Ensure data is sorted by date
            df = df.sort_values("date")
            
            return df
            
        elif response.status_code == 429:
            # Rate limit exceeded, wait and try again
            print("Rate limit exceeded. Waiting before retrying...")
            time.sleep(60)  # Wait for 60 seconds
            return fetch_crypto_data(crypto_id, start_date, end_date)
            
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return None
        
def get_daily_data(df):
    """
    Resample the data to get daily closing prices.
    
    Args:
        df (pandas.DataFrame): Dataframe with historical price data
        
    Returns:
        pandas.DataFrame: Dataframe with daily price data
    """
    # Set date as index if it's not already
    if 'date' in df.columns:
        df = df.set_index('date')
    
    # Resample to daily frequency
    daily_data = df.resample('D').agg({
        'price': 'last',
        'market_cap': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Reset index to make date a column again
    daily_data = daily_data.reset_index()
    
    return daily_data
