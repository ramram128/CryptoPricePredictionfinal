import pandas as pd
import numpy as np
from data_utils import get_daily_data

def calculate_moving_averages(df, windows=[7, 14, 30]):
    """
    Calculate moving averages for the given windows.
    
    Args:
        df (pandas.DataFrame): Dataframe with price data
        windows (list): List of window sizes for moving averages
        
    Returns:
        pandas.DataFrame: Dataframe with moving averages added
    """
    for window in windows:
        df[f'ma_{window}'] = df['price'].rolling(window=window).mean()
    return df

def calculate_rsi(df, window=14):
    """
    Calculate the Relative Strength Index (RSI).
    
    Args:
        df (pandas.DataFrame): Dataframe with price data
        window (int): Window size for RSI calculation
        
    Returns:
        pandas.DataFrame: Dataframe with RSI added
    """
    # Calculate price changes
    delta = df['price'].diff()
    
    # Separate gains and losses
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = abs(loss)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    df['rsi'] = rsi
    return df

def prepare_features(df, use_moving_avg=True, use_volume=True, use_rsi=True):
    """
    Prepare features for machine learning models.
    
    Args:
        df (pandas.DataFrame): Dataframe with historical price data
        use_moving_avg (bool): Whether to include moving averages as features
        use_volume (bool): Whether to include volume as a feature
        use_rsi (bool): Whether to include RSI as a feature
        
    Returns:
        tuple: (X, y, dates, feature_names)
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target vector
            dates (list): List of dates
            feature_names (list): List of feature names
    """
    # Make sure we have daily data
    df = get_daily_data(df)
    
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Calculate additional features
    if use_moving_avg:
        data = calculate_moving_averages(data)
    
    if use_rsi:
        data = calculate_rsi(data)
    
    # Add lag features (previous day's price)
    data['price_lag1'] = data['price'].shift(1)
    data['price_lag2'] = data['price'].shift(2)
    data['price_lag3'] = data['price'].shift(3)
    
    # Drop rows with NaN values (due to lag features and moving averages)
    data = data.dropna()
    
    # Initialize feature list
    features = ['price_lag1', 'price_lag2', 'price_lag3']
    
    # Add moving averages if requested
    if use_moving_avg:
        features.extend(['ma_7', 'ma_14', 'ma_30'])
    
    # Add volume if requested
    if use_volume:
        features.append('volume')
    
    # Add RSI if requested
    if use_rsi:
        features.append('rsi')
    
    # Create feature matrix X and target vector y
    X = data[features].values
    y = data['price'].values
    dates = data['date'].tolist()
    
    return X, y, dates, features
