import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

from data_utils import fetch_crypto_data
from feature_engineering import prepare_features
from model_utils import train_linear_regression, train_random_forest, evaluate_model, make_prediction

# Page configuration
st.set_page_config(
    page_title="Crypto Price Predictor",
    page_icon="📈",
    layout="wide"
)

# App title and description
st.title("Cryptocurrency Price Prediction")
st.write("""
This application predicts cryptocurrency prices using machine learning models.
Select a cryptocurrency, date range, and model to see the prediction for the next day's price!
""")

# Sidebar for user inputs
st.sidebar.header("Settings")

# Cryptocurrency selection
crypto_options = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "Litecoin": "litecoin",
    "Ripple": "ripple",
    "Cardano": "cardano"
}
selected_crypto = st.sidebar.selectbox("Select Cryptocurrency", list(crypto_options.keys()))
crypto_id = crypto_options[selected_crypto]

# Date range selection
end_date = datetime.now().date()
start_date_default = end_date - timedelta(days=365)  # Default to last year
start_date = st.sidebar.date_input("Start Date", start_date_default)
end_date = st.sidebar.date_input("End Date", end_date)

if start_date >= end_date:
    st.error("Error: Start date must be before end date.")
    st.stop()

# Model selection
model_options = ["Linear Regression", "Random Forest"]
selected_model = st.sidebar.selectbox("Select Model", model_options)

# Feature selection
st.sidebar.subheader("Feature Selection")
use_moving_avg = st.sidebar.checkbox("Use Moving Averages", True)
use_volume = st.sidebar.checkbox("Use Volume Data", True)
use_rsi = st.sidebar.checkbox("Use RSI (Relative Strength Index)", True)

# Train-test split ratio
test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20) / 100

# Add a button to trigger the prediction
predict_button = st.sidebar.button("Generate Prediction")

# Main content
if predict_button:
    # Show loading spinner
    with st.spinner("Fetching data and preparing models..."):
        # Fetch data
        try:
            df = fetch_crypto_data(crypto_id, start_date, end_date)
            
            if df is None or df.empty:
                st.error("Failed to fetch data. Please try again or select a different cryptocurrency.")
                st.stop()
                
            # Display raw data preview
            st.subheader("Raw Data Preview")
            st.dataframe(df.head())
            
            # Prepare features
            X, y, dates, feature_names = prepare_features(
                df, 
                use_moving_avg=use_moving_avg, 
                use_volume=use_volume, 
                use_rsi=use_rsi
            )
            
            # Split data into training and test sets
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            dates_train, dates_test = dates[:split_idx], dates[split_idx:]
            
            # Train the selected model
            if selected_model == "Linear Regression":
                model, y_pred = train_linear_regression(X_train, y_train, X_test)
                model_name = "Linear Regression"
            else:
                model, y_pred = train_random_forest(X_train, y_train, X_test)
                model_name = "Random Forest"
            
            # Evaluate model
            metrics = evaluate_model(y_test, y_pred)
            
            # Make prediction for the next day
            next_day_pred = make_prediction(model, X, y, feature_names, df)
            
            # Display evaluation metrics
            st.subheader("Model Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Absolute Error (MAE)", f"${metrics['mae']:.2f}")
            with col2:
                st.metric("Root Mean Squared Error (RMSE)", f"${metrics['rmse']:.2f}")
            with col3:
                st.metric("R² Score", f"{metrics['r2']:.4f}")
            
            # Display prediction for tomorrow
            st.subheader("Next Day Prediction")
            current_price = df['price'].iloc[-1]
            price_change = next_day_pred - current_price
            percentage_change = (price_change / current_price) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    f"Predicted {selected_crypto} Price Tomorrow", 
                    f"${next_day_pred:.2f}", 
                    f"{percentage_change:.2f}%"
                )
            with col2:
                st.metric(
                    f"Current {selected_crypto} Price", 
                    f"${current_price:.2f}"
                )
            
            # Visualize results
            st.subheader("Price Prediction vs Actual")
            
            # Create a dataframe for visualization
            results_df = pd.DataFrame({
                'Date': dates_test,
                'Actual': y_test,
                'Predicted': y_pred
            })
            
            # Create a Plotly figure
            fig = go.Figure()
            
            # Add actual prices
            fig.add_trace(go.Scatter(
                x=results_df['Date'],
                y=results_df['Actual'],
                mode='lines',
                name='Actual Price',
                line=dict(color='blue')
            ))
            
            # Add predicted prices
            fig.add_trace(go.Scatter(
                x=results_df['Date'],
                y=results_df['Predicted'],
                mode='lines',
                name='Predicted Price',
                line=dict(color='red')
            ))
            
            # Add next day prediction
            next_date = dates[-1] + timedelta(days=1)
            fig.add_trace(go.Scatter(
                x=[next_date],
                y=[next_day_pred],
                mode='markers',
                marker=dict(size=10, color='green'),
                name='Next Day Prediction'
            ))
            
            # Update layout
            fig.update_layout(
                title=f'{selected_crypto} Price Prediction using {model_name}',
                xaxis_title='Date',
                yaxis_title='Price (USD)',
                legend=dict(x=0, y=1, traceorder='normal'),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display historical price chart
            st.subheader(f"Historical {selected_crypto} Prices")
            
            # Create a historical price dataframe
            hist_df = pd.DataFrame({
                'Date': dates,
                'Price': y
            })
            
            # Create a Plotly figure for historical data
            hist_fig = go.Figure()
            
            # Add historical prices
            hist_fig.add_trace(go.Scatter(
                x=hist_df['Date'],
                y=hist_df['Price'],
                mode='lines',
                name=f'{selected_crypto} Price',
                line=dict(color='blue')
            ))
            
            # Update layout
            hist_fig.update_layout(
                title=f'Historical {selected_crypto} Prices',
                xaxis_title='Date',
                yaxis_title='Price (USD)',
                height=400
            )
            
            st.plotly_chart(hist_fig, use_container_width=True)
            
            # Display feature importance if Random Forest
            if selected_model == "Random Forest":
                st.subheader("Feature Importance")
                
                # Get feature importance
                feature_importance = model.feature_importances_
                
                # Create a dataframe for feature importance
                fi_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': feature_importance
                }).sort_values(by='Importance', ascending=False)
                
                # Plot feature importance
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(fi_df['Feature'], fi_df['Importance'])
                ax.set_xlabel('Importance')
                ax.set_title('Feature Importance (Random Forest)')
                plt.tight_layout()
                
                st.pyplot(fig)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
else:
    # Display instructions when the app is first loaded
    st.info("👈 Configure the settings on the sidebar and click 'Generate Prediction' to start!")
    
    # Display some information about the app
    st.subheader("About this App")
    st.write("""
    This application uses machine learning to predict cryptocurrency prices based on historical data. 
    
    **How it works:**
    1. Historical data is fetched from CoinGecko API
    2. Features are engineered from the price and volume data
    3. Machine learning models are trained on the historical data
    4. The models predict the next day's closing price
    
    **Available Models:**
    - **Linear Regression**: A simple model that assumes a linear relationship between features and price
    - **Random Forest**: An ensemble model that combines multiple decision trees for better predictions
    
    **Features Used:**
    - Previous days' prices
    - Moving averages (7-day, 14-day, 30-day)
    - Trading volume
    - Technical indicators like RSI (Relative Strength Index)
    """)

# Footer
st.markdown("---")
st.markdown("<center>Developed with ❤️ using Streamlit</center>", unsafe_allow_html=True)
