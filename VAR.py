import yfinance as yf
import numpy as np
from scipy.stats import norm
tickers = ['MSFT', 'AAPL']
weights = [0.6, 0.4]  
start_date = "2022-09-18"
end_date = "2023-09-18"
data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
returns = data.pct_change().dropna()
portfolio_returns = np.dot(returns, weights)
portfolio_std_dev = np.std(portfolio_returns)
confidence_level = 0.95  
time_horizon = 1  
z_score = norm.ppf(confidence_level)
var = portfolio_std_dev * z_score * np.sqrt(time_horizon)

print(f"Portfolio VaR (Variance-Covariance Approach): {var:.2f} USD")
