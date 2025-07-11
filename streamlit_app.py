import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go
from datetime import datetime

st.set_page_config(page_title="Bitcoin Price Forecast (Custom Date Range)", layout="wide")

st.title("📈 Bitcoin Price Forecast with Custom Date Range & Projection Horizon")

# Upload CSV
uploaded_file = st.file_uploader("Upload your BTC Price CSV", type=['csv'])

if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])

    # Select data field dynamically
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    selected_field = st.sidebar.selectbox("Select Data Field for Forecast", numeric_columns, index=0)

    df = df[['Date', selected_field]].rename(columns={'Date': 'ds', selected_field: 'y'})
    df.dropna(inplace=True)

    # Sidebar: Date Range Selection
    st.sidebar.header("🔎 Select Historical Date Range for Forecast")
    min_date = df['ds'].min().date()
    max_date = df['ds'].max().date()

    start_date = st.sidebar.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.sidebar.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

    # Sidebar: Projection End Date
    st.sidebar.header("🛠 Select Forecast Horizon")
    projection_end_date = st.sidebar.date_input("Forecast Until Date", min_value=end_date, value=max_date)

    # Filter Data
    filtered_df = df[(df['ds'].dt.date >= start_date) & (df['ds'].dt.date <= end_date)]

    st.subheader(f"📊 BTC Historical {selected_field} from {start_date} to {end_date}")
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(x=filtered_df['ds'], y=filtered_df['y'], mode='lines', name=f'Historical {selected_field}'))
    fig_raw.update_layout(title=f'BTC Historical {selected_field}', xaxis_title='Date', yaxis_title=selected_field)
    st.plotly_chart(fig_raw, use_container_width=True)

    # Forecast Button
    if st.button("🚀 Start Forecast"):
        if filtered_df.empty:
            st.error("❌ Selected date range has no data. Please adjust the dates.")
        else:
            st.success(f"Running Forecast using {selected_field} from {start_date} to {end_date}...")

            # Train Prophet only on selected date range
            model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
            model.fit(filtered_df)

            # Set forecast horizon
            last_date = filtered_df['ds'].max().date()
            delta_days = (projection_end_date - last_date).days
            delta_days = max(delta_days, 0)

            future = model.make_future_dataframe(periods=delta_days)
            forecast = model.predict(future)

            # Merge Historical + Forecast
            merged = pd.merge(filtered_df, forecast[['ds', 'yhat']], on='ds', how='outer')

            # Plot Combined Chart
            st.subheader(f"🔮 BTC {selected_field} Forecast (Based on {start_date} to {end_date}, Projected to {projection_end_date})")
            fig_overlap = go.Figure()

            fig_overlap.add_trace(go.Scatter(
                x=merged['ds'], y=merged['y'],
                mode='lines', name=f'Actual {selected_field}', line=dict(color='blue')
            ))

            fig_overlap.add_trace(go.Scatter(
                x=merged['ds'], y=merged['yhat'],
                mode='lines', name='Forecast', line=dict(color='orange', dash='dot')
            ))

            fig_overlap.update_layout(title=f'BTC {selected_field}: Actual vs. Forecast (Data: {start_date}–{end_date}, Forecast: until {projection_end_date})',
                                      xaxis_title='Date', yaxis_title=selected_field,
                                      legend=dict(x=0, y=1))

            st.plotly_chart(fig_overlap, use_container_width=True)

            # Forecast Table & Download
            st.subheader(f"📄 Forecast Data (Up to {projection_end_date})")
            forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            st.dataframe(forecast_df.tail(365))

            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Forecast CSV", data=csv, file_name=f'btc_forecast_{start_date}_to_{end_date}_until_{projection_end_date}.csv', mime='text/csv')

else:
    st.info("👆 Please upload a CSV file to get started.")