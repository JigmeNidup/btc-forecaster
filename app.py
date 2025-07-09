import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# 1. Load the BTC Price CSV
# Replace with your actual CSV filename
csv_file = 'btc_prices.csv'
df = pd.read_csv(csv_file)

# 2. Preprocess Data
# Ensure correct date parsing and select necessary columns
df['Date'] = pd.to_datetime(df['Date'])
df = df[['Date', 'Close**']].rename(columns={'Date': 'ds', 'Close**': 'y'})

# Optional: Remove rows with missing values
df.dropna(inplace=True)

# 3. Initialize and Train Prophet Model
model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
model.fit(df)

# 4. Create Future DataFrame (Predicting until end of 2028)
future = model.make_future_dataframe(periods=365 * 3)  # Next 3 years (2026, 2027, 2028)

# 5. Forecast
forecast = model.predict(future)

# 6. Plot Forecast
model.plot(forecast)
plt.title('Bitcoin Price Forecast until 2028')
plt.xlabel('Date')
plt.ylabel('BTC Price (USD)')
plt.show()

# 7. Optional: Save forecast to CSV
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('btc_forecast_2028.csv', index=False)

print("✅ Forecast complete. Plot displayed and results saved to 'btc_forecast_2028.csv'.")


# libs
# pip install prophet pandas matplotlib
