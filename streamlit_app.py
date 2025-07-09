import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

st.set_page_config(page_title="Bitcoin Price Forecast", layout="wide")

# Title
st.title("📈 Bitcoin Price Forecast (2011–2028)")

# File uploader
uploaded_file = st.file_uploader("Upload your BTC Price CSV", type=['csv'])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Close**']].rename(columns={'Date': 'ds', 'Price': 'y'})
    df.dropna(inplace=True)

    # Display historical data
    st.subheader("📊 Historical BTC Prices")
    fig1 = px.line(df, x='ds', y='y', title='Historical BTC Prices')
    st.plotly_chart(fig1, use_container_width=True)

    # Prophet Forecast
    model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df)

    future = model.make_future_dataframe(periods=365 * 3)  # Predict to 2028
    forecast = model.predict(future)

    # Plot forecast
    st.subheader("🔮 Forecasted BTC Prices (Up to 2028)")
    fig2 = px.line(forecast, x='ds', y='yhat', labels={'yhat': 'Predicted Price'}, title='BTC Forecast')
    st.plotly_chart(fig2, use_container_width=True)

    # Show forecast DataFrame
    st.subheader("📄 Forecast Data")
    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    st.dataframe(forecast_df.tail(365))

    # Download button
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Forecast CSV", data=csv, file_name='btc_forecast_2028.csv', mime='text/csv')
else:
    st.info("👆 Please upload a CSV file to get started.")

# libs
# pip install streamlit prophet pandas matplotlib plotly


# run 
# streamlit run btc_forecast_app.py
