import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import timedelta

def train_linear_regression(X_train, y_train, X_test):
    """
    Train a Linear Regression model.
    
    Args:
        X_train (numpy.ndarray): Training feature matrix
        y_train (numpy.ndarray): Training target vector
        X_test (numpy.ndarray): Test feature matrix
        
    Returns:
        tuple: (model, y_pred)
            model: Trained Linear Regression model
            y_pred (numpy.ndarray): Predictions on test data
    """
    # Initialize model
    model = LinearRegression()
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    return model, y_pred

def train_random_forest(X_train, y_train, X_test):
    """
    Train a Random Forest Regression model.
    
    Args:
        X_train (numpy.ndarray): Training feature matrix
        y_train (numpy.ndarray): Training target vector
        X_test (numpy.ndarray): Test feature matrix
        
    Returns:
        tuple: (model, y_pred)
            model: Trained Random Forest model
            y_pred (numpy.ndarray): Predictions on test data
    """
    # Initialize model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    return model, y_pred

def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance using various metrics.
    
    Args:
        y_true (numpy.ndarray): True target values
        y_pred (numpy.ndarray): Predicted target values
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

def make_prediction(model, X, y, feature_names, df):
    """
    Make a prediction for the next day.
    
    Args:
        model: Trained machine learning model
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target vector
        feature_names (list): List of feature names
        df (pandas.DataFrame): Original dataframe with price data
        
    Returns:
        float: Predicted price for the next day
    """
    # Get the latest data
    latest_data = df.iloc[-1]
    latest_price = latest_data['price']
    
    # Create a feature vector for the next day
    next_day_features = {}
    
    # Price lag features
    next_day_features['price_lag1'] = latest_price
    next_day_features['price_lag2'] = df.iloc[-2]['price'] if len(df) > 1 else latest_price
    next_day_features['price_lag3'] = df.iloc[-3]['price'] if len(df) > 2 else latest_price
    
    # Moving average features
    if 'ma_7' in feature_names:
        next_day_features['ma_7'] = df['price'].tail(7).mean()
    
    if 'ma_14' in feature_names:
        next_day_features['ma_14'] = df['price'].tail(14).mean()
    
    if 'ma_30' in feature_names:
        next_day_features['ma_30'] = df['price'].tail(30).mean()
    
    # Volume feature
    if 'volume' in feature_names:
        next_day_features['volume'] = latest_data['volume']
    
    # RSI feature
    if 'rsi' in feature_names:
        next_day_features['rsi'] = latest_data['rsi'] if 'rsi' in latest_data else 50  # Default to neutral RSI if not available
    
    # Create feature vector in the same order as during training
    next_day_X = [next_day_features[feature] for feature in feature_names]
    
    # Make prediction
    prediction = model.predict(np.array([next_day_X]))[0]
    
    return prediction
