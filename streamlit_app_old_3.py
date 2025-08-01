import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objs as go
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Bitcoin Price Forecast Dashboard", layout="wide")
st.title("📈 Bitcoin Price Forecast Dashboard")

uploaded_file = st.file_uploader("Upload your CSV (Date + Numeric Fields)", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df)  # Show full CSV

    date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower()]
    if not date_columns:
        st.error("❌ No valid Date column found.")
    else:
        date_col = date_columns[0]
        df[date_col] = pd.to_datetime(df[date_col])

        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != date_col]
        if not numeric_cols:
            st.error("❌ No numeric fields to forecast.")
        else:
            target_field = st.sidebar.selectbox("Select field to forecast:", numeric_cols)

            min_date = df[date_col].min().date()
            max_date = df[date_col].max().date()

            start_date = st.sidebar.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
            end_date = st.sidebar.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

            projection_date = st.sidebar.date_input("Projection Date (Year)", value=datetime(2028, 12, 31), min_value=max_date)

            # Model descriptions dictionary
            model_descriptions = {
                "Prophet": """
**Prophet** by Facebook is a robust model for forecasting time series data with trends and seasonality.  
It handles missing values and outliers effectively and is widely used for business forecasts.
""",
                "LSTM": """
**LSTM (Long Short-Term Memory)** is a deep learning model that captures complex temporal dependencies.  
It is powerful for non-linear time series forecasting, especially when patterns are not obvious.
""",
                "SARIMA": """
**SARIMA (Seasonal ARIMA)** is a traditional statistical method that models seasonality, trend, and autocorrelation.  
It works best with stable patterns and cyclical behavior in time series data.
""",
                "Monte Carlo": """
**Monte Carlo Simulation** generates multiple future paths using random sampling based on historical volatility.  
It provides a probabilistic range of possible outcomes rather than a single forecast.
""", 
                "Moving Averages (MAs)": """
**Moving Averages (50-day & 200-day)** are simple technical indicators used to smooth historical price data.  
They are not predictive models but can help identify trend directions and crossover signals.
"""
            }

            model_choice = st.sidebar.selectbox("Forecasting Model", list(model_descriptions.keys()))

            with st.sidebar.expander("ℹ️ Info about Selected Model"):
                st.markdown(model_descriptions[model_choice])

            simulations = 400
            if model_choice == "Monte Carlo":
                simulations = st.sidebar.number_input("Number of Simulations", min_value=100, max_value=10000, value=400, step=100)

            filtered_df = df[(df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)][[date_col, target_field]].rename(columns={date_col: 'ds', target_field: 'y'}).dropna()

            st.subheader("📊 Selected Data")
            fig_selected = go.Figure()
            fig_selected.add_trace(go.Scatter(x=filtered_df['ds'], y=filtered_df['y'], name='Selected Historical Data'))
            fig_selected.update_layout(title='Selected Data Graph', xaxis_title='Date', yaxis_title=target_field)
            st.plotly_chart(fig_selected, use_container_width=True)

            if st.button("🚀 Start Forecast"):
                with st.spinner('Forecasting in progress...'):
                    last_date = filtered_df['ds'].max().date()
                    forecast_end = projection_date
                    future_days = max((forecast_end - last_date).days, 0)

                    if model_choice == "Prophet":
                        # Define halving dates
                        halving_dates = pd.DataFrame({
                            'holiday': 'halving',
                            'ds': pd.to_datetime([
                                '2012-11-28',
                                '2016-07-09',
                                '2020-05-11',
                                # Add expected future halvings if you want, e.g.:
                                # '2024-04-20'
                            ]),
                            'lower_window': -30,  # 30 days before halving date
                            'upper_window': 30    # 30 days after halving date
                        })

                        model = Prophet(holidays=halving_dates)
                        model.fit(filtered_df)
                        future = model.make_future_dataframe(periods=future_days)
                        forecast = model.predict(future)
                        combined = pd.merge(filtered_df, forecast[['ds', 'yhat']], on='ds', how='right')
                        combined['yhat'].fillna(method='ffill', inplace=True)

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=combined['ds'], y=combined['y'], name='Actual'))
                        fig.add_trace(go.Scatter(x=combined['ds'], y=combined['yhat'], name='Forecast', line=dict(color='red')))
                        st.plotly_chart(fig, use_container_width=True)

                        st.subheader("📋 Forecasted Results Table")
                        st.dataframe(combined[['ds', 'yhat']].tail(365))
                        
                        csv = combined.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Forecast CSV", data=csv, file_name=f'btc_forecast_{start_date}_to_{end_date}_until_{forecast_end}.csv', mime='text/csv')

                    elif model_choice == "LSTM":
                        scaler = MinMaxScaler()
                        scaled_y = scaler.fit_transform(filtered_df[['y']])
                        seq_len = 30
                        X, y_seq = [], []
                        for i in range(seq_len, len(scaled_y)):
                            X.append(scaled_y[i - seq_len:i, 0])
                            y_seq.append(scaled_y[i, 0])
                        X, y_seq = np.array(X), np.array(y_seq)
                        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

                        model = Sequential([
                            Input(shape=(seq_len, 1)),
                            LSTM(50, return_sequences=True),
                            LSTM(50),
                            Dense(1)
                        ])
                        model.compile(optimizer='adam', loss='mean_squared_error')
                        model.fit(X, y_seq, epochs=10, batch_size=16, verbose=0)

                        predictions = []
                        for i in range(seq_len, len(scaled_y)):
                            seq_input = scaled_y[i - seq_len:i].reshape(1, seq_len, 1)
                            pred = model.predict(seq_input, verbose=0)[0][0]
                            predictions.append(pred)

                        last_seq = scaled_y[-seq_len:].flatten().tolist()
                        for _ in range(future_days):
                            seq_input = np.array(last_seq[-seq_len:]).reshape(1, seq_len, 1)
                            pred = model.predict(seq_input, verbose=0)[0][0]
                            last_seq.append(pred)
                            predictions.append(pred)

                        all_preds = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
                        full_dates = pd.date_range(start=filtered_df['ds'].iloc[seq_len], periods=len(all_preds), freq='D')

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=filtered_df['ds'], y=filtered_df['y'], name='Actual'))
                        fig.add_trace(go.Scatter(x=full_dates, y=all_preds, name='Forecast', line=dict(color='red')))
                        st.plotly_chart(fig, use_container_width=True)

                        st.subheader("📋 Forecasted Results Table")
                        forecast_df = pd.DataFrame({'ds': full_dates, 'yhat': all_preds})
                        st.dataframe(forecast_df.tail(365))

                    elif model_choice == "SARIMA":
                        sarima_model = SARIMAX(filtered_df['y'], order=(1,1,1), seasonal_order=(1,1,1,12))
                        sarima_fit = sarima_model.fit(disp=False)
                        total_steps = len(filtered_df) + future_days
                        pred = sarima_fit.get_prediction(start=0, end=total_steps-1)
                        pred_mean = pred.predicted_mean
                        full_dates = pd.date_range(start=filtered_df['ds'].iloc[0], periods=total_steps, freq='D')

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=filtered_df['ds'], y=filtered_df['y'], name='Actual'))
                        fig.add_trace(go.Scatter(x=full_dates, y=pred_mean, name='Forecast', line=dict(color='red')))
                        st.plotly_chart(fig, use_container_width=True)

                        st.subheader("📋 Forecasted Results Table")
                        forecast_df = pd.DataFrame({'ds': full_dates, 'yhat': pred_mean.values})
                        st.dataframe(forecast_df.tail(365))

                    elif model_choice == "Monte Carlo":
                        returns = filtered_df['y'].pct_change().dropna()
                        mu, sigma = returns.mean(), returns.std()
                        total_days = len(filtered_df) + future_days
                        sim = np.zeros((total_days, simulations))
                        for i in range(simulations):
                            path = [filtered_df['y'].iloc[0]]
                            path += list(filtered_df['y'].iloc[1:].values)
                            for _ in range(future_days):
                                path.append(path[-1] * np.exp(np.random.normal(mu, sigma)))
                            sim[:, i] = path

                        median_forecast = np.median(sim, axis=1)
                        full_dates = pd.date_range(start=filtered_df['ds'].iloc[0], periods=total_days, freq='D')

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=filtered_df['ds'], y=filtered_df['y'], name='Actual'))
                        for i in range(0, simulations, max(1, simulations // 50)):
                            fig.add_trace(go.Scatter(
                                x=full_dates, y=sim[:, i], mode='lines',
                                line=dict(width=0.5), name=f'Sim {i}', opacity=0.2, visible='legendonly'))
                        fig.add_trace(go.Scatter(
                            x=full_dates, y=median_forecast, name='Median Forecast',
                            line=dict(color='red', dash='dot')))
                        st.plotly_chart(fig, use_container_width=True)

                        st.subheader("📋 Forecasted Results Table")
                        forecast_df = pd.DataFrame({'ds': full_dates, 'yhat': median_forecast})
                        st.dataframe(forecast_df.tail(365))
                        
                    elif model_choice == "Moving Averages (MAs)":
                        df_ma = filtered_df.copy()
                        df_ma['MA_50'] = df_ma['y'].rolling(window=50).mean()
                        df_ma['MA_200'] = df_ma['y'].rolling(window=200).mean()

                        # Calculate the last slope (day-to-day difference) of MA_50 and MA_200
                        ma_50_slope = df_ma['MA_50'].diff().dropna().iloc[-1]
                        ma_200_slope = df_ma['MA_200'].diff().dropna().iloc[-1]

                        # Number of days to forecast
                        last_date = df_ma['ds'].max()
                        future_days = max((projection_date - last_date.date()).days, 0)
                        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='D')

                        # Extend MA values into the future using the last slope (linear extrapolation)
                        last_ma_50 = df_ma['MA_50'].iloc[-1]
                        last_ma_200 = df_ma['MA_200'].iloc[-1]

                        forecast_ma_50 = [last_ma_50 + ma_50_slope * (i+1) for i in range(future_days)]
                        forecast_ma_200 = [last_ma_200 + ma_200_slope * (i+1) for i in range(future_days)]

                        # For forecasted price, take average of both MAs
                        forecast_price = [(f50 + f200) / 2 for f50, f200 in zip(forecast_ma_50, forecast_ma_200)]

                        # Combine historical and forecast data for plotting
                        df_forecast = pd.DataFrame({
                            'ds': future_dates,
                            'Forecast_MA_50': forecast_ma_50,
                            'Forecast_MA_200': forecast_ma_200,
                            'Forecast_Price': forecast_price
                        })

                        # Plot historical actual and MAs
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df_ma['ds'], y=df_ma['y'], name='Actual', line=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=df_ma['ds'], y=df_ma['MA_50'], name='50-Day MA', line=dict(color='orange')))
                        fig.add_trace(go.Scatter(x=df_ma['ds'], y=df_ma['MA_200'], name='200-Day MA', line=dict(color='green')))

                        # Plot forecasted MAs and price
                        fig.add_trace(go.Scatter(x=df_forecast['ds'], y=df_forecast['Forecast_MA_50'], 
                                                name='Forecast 50-Day MA', line=dict(color='orange', dash='dot')))
                        fig.add_trace(go.Scatter(x=df_forecast['ds'], y=df_forecast['Forecast_MA_200'], 
                                                name='Forecast 200-Day MA', line=dict(color='green', dash='dot')))
                        fig.add_trace(go.Scatter(x=df_forecast['ds'], y=df_forecast['Forecast_Price'], 
                                                name='Forecast Price (Avg MA)', line=dict(color='red', dash='dash')))

                        fig.update_layout(title=f'50 & 200-Day Moving Averages with Forecast till {projection_date}', 
                                        xaxis_title='Date', yaxis_title=target_field)
                        st.plotly_chart(fig, use_container_width=True)

                        # Show forecast table
                        st.subheader("📋 Forecasted Moving Averages & Price")
                        st.dataframe(df_forecast[['ds', 'Forecast_MA_50', 'Forecast_MA_200', 'Forecast_Price']].tail(365))


else:
    st.info("📤 Please upload a CSV file to begin.")
