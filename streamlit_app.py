import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Bitcoin Price Forecast (2011-2028)", layout="wide")

# Title
st.title("📈 Bitcoin Price Forecast with Custom Date Selection")

# Upload CSV
uploaded_file = st.file_uploader("Upload your BTC Price CSV", type=['csv'])

if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Price']].rename(columns={'Date': 'ds', 'Price': 'y'})
    df.dropna(inplace=True)

    # Select Date Range for Historical Display
    st.sidebar.header("🔎 Select Historical Date Range")
    min_date = df['ds'].min().date()
    max_date = df['ds'].max().date()

    start_date = st.sidebar.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.sidebar.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

    filtered_df = df[(df['ds'].dt.date >= start_date) & (df['ds'].dt.date <= end_date)]

    # Display Raw Historical Chart
    st.subheader(f"📊 BTC Historical Prices from {start_date} to {end_date}")
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(x=filtered_df['ds'], y=filtered_df['y'], mode='lines', name='Historical Price'))
    fig_raw.update_layout(title='BTC Historical Prices', xaxis_title='Date', yaxis_title='Price (USD)')
    st.plotly_chart(fig_raw, use_container_width=True)

    # Select Projection End Year
    st.sidebar.header("🛠 Select Projection End Year")
    projection_year = st.sidebar.selectbox("Forecast Until Year", [2026, 2027, 2028, 2029, 2030], index=2)

    # Build Prophet Model
    model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df)

    # Compute Future Periods
    last_date = df['ds'].max().date()
    forecast_end_date = datetime(projection_year, 12, 31).date()
    delta_days = (forecast_end_date - last_date).days
    if delta_days < 0:
        delta_days = 0  # No future prediction if date is in the past

    future = model.make_future_dataframe(periods=delta_days)
    forecast = model.predict(future)

    # Merge for Overlap Plot
    merged = pd.merge(df, forecast[['ds', 'yhat']], on='ds', how='outer')

    # Plot Overlap
    st.subheader(f"🔮 BTC Actual vs. Forecast (Up to {projection_year})")
    fig_overlap = go.Figure()

    fig_overlap.add_trace(go.Scatter(
        x=merged['ds'], y=merged['y'],
        mode='lines', name='Actual Price', line=dict(color='blue')
    ))

    fig_overlap.add_trace(go.Scatter(
        x=merged['ds'], y=merged['yhat'],
        mode='lines', name='Forecast Price', line=dict(color='orange', dash='dot')
    ))

    fig_overlap.update_layout(title=f'BTC Price: Actual vs. Forecast (2011–{projection_year})',
                              xaxis_title='Date', yaxis_title='Price (USD)',
                              legend=dict(x=0, y=1))

    st.plotly_chart(fig_overlap, use_container_width=True)

    # Display Forecast Table & Download
    st.subheader(f"📄 Forecast Data (Up to {projection_year})")
    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    st.dataframe(forecast_df.tail(365))

    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Forecast CSV", data=csv, file_name=f'btc_forecast_until_{projection_year}.csv', mime='text/csv')

else:
    st.info("👆 Please upload a CSV file to get started.")
