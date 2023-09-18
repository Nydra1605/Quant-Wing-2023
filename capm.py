import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

# Define the ticker symbols
stock_symbol = "RELIANCE.NS"   # Ticker for Infosys Limited on BSE (in INR)
index_symbol = "^NSEI"     # NIFTY 50 index on NSE (in INR)

# Define the timeframe
start_date = "2022-09-18"
end_date = "2023-09-18"

# Download historical data for the stock and the NIFTY 50 index
stock_data = yf.download(stock_symbol, start=start_date, end=end_date, progress=False)
index_data = yf.download(index_symbol, start=start_date, end=end_date, progress=False)

# Extract the closing prices
stock_close = stock_data["Close"]
index_close = index_data["Close"]

# Calculate cumulative returns
stock_cumulative_return = (stock_close / stock_close.iloc[0] - 1) * 100
index_cumulative_return = (index_close / index_close.iloc[0] - 1) * 100

# Plot the closing prices
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(stock_close, label=f"{stock_symbol} Closing Price (INR)", color='b')
plt.plot(index_close, label="NIFTY 50 Closing Price (INR)", color='g')
plt.title("Closing Prices Comparison")
plt.ylabel("Price (INR)")
plt.legend()

# Plot the cumulative returns
plt.subplot(2, 1, 2)
plt.plot(stock_cumulative_return, label=f"{stock_symbol} Cumulative Return", color='b')
plt.plot(index_cumulative_return, label="NIFTY 50 Cumulative Return", color='g')
plt.title("Cumulative Returns Comparison")
plt.ylabel("Cumulative Return (%)")
plt.legend()

plt.tight_layout()
plt.show()

# Calculate daily returns
stock_daily_return = stock_data["Adj Close"].pct_change()
index_daily_return = index_data["Adj Close"].pct_change()

# Drop missing values
stock_daily_return = stock_daily_return.dropna()
index_daily_return = index_daily_return.dropna()

# Perform linear regression
X = np.array(index_daily_return).reshape(-1, 1)
y = np.array(stock_daily_return).reshape(-1, 1)

model = LinearRegression().fit(X, y)

beta = model.coef_[0][0]
alpha = model.intercept_[0]
r_value = model.score(X, y)

print("Linear Regression Results:")
print("Beta (Market Sensitivity):", beta)
print("Alpha (Intercept):", alpha)
print("R-squared Value:", r_value)

# Calculate expected return
risk_free_rate = 0.06  # 6% risk-free rate (assumed)
expected_return = risk_free_rate + beta * (index_daily_return.mean() * 252 - risk_free_rate)

print("Expected Return based on CAPM:", expected_return)
