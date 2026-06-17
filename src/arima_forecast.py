import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("data/JPM.csv")

# Clean
df = df.iloc[1:].copy()
df.rename(columns={"Price": "Date"}, inplace=True)

df["Date"] = pd.to_datetime(df["Date"])

for col in ["Close", "High", "Low", "Open", "Volume"]:
    df[col] = pd.to_numeric(df[col])

# Use Close price
series = df["Close"]

# Train-Test Split
train_size = int(len(series) * 0.8)

train = series[:train_size]
test = series[train_size:]

# Train ARIMA
model = ARIMA(train, order=(2,1,1))
result = model.fit()

# Forecast
forecast = result.forecast(steps=len(test))

# Metrics
mae = mean_absolute_error(test, forecast)
rmse = mean_squared_error(test, forecast) ** 0.5

print("MAE :", mae)
print("RMSE:", rmse)

# Plot
plt.figure(figsize=(12,6))

plt.plot(test.index, test,
         label="Actual")

plt.plot(test.index, forecast,
         label="Forecast")

plt.title("ARIMA Forecast - JPM")
plt.legend()

plt.show()