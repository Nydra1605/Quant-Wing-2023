import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


ticker = "AAPL"  
start_date = "2021-01-01"
end_date = "2023-01-06"
data = yf.download(ticker, start=start_date, end=end_date, progress=False)

data["Return"] = data["Close"].pct_change()  
data.dropna(inplace=True)


data["SMA_5"] = data["Close"].rolling(window=5).mean()
data["SMA_20"] = data["Close"].rolling(window=20).mean()


X = data[["SMA_5", "SMA_20"]]
y = (data["Return"] > 0).astype(int) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)


y_pred_train = pipeline.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)

y_pred_test = pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


data["Predicted_Return"] = pipeline.predict(X)
data["Signal"] = data["Predicted_Return"].diff()
data.loc[data["Signal"] > 0, "Position"] = 1  
data.loc[data["Signal"] < 0, "Position"] = -1  
data["Position"].fillna(0, inplace=True) 


data["Strategy_Return"] = data["Position"] * data["Return"]
cumulative_returns = (data["Strategy_Return"] + 1).cumprod()

plt.figure(figsize=(10, 6))
plt.plot(data.index, cumulative_returns)
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.title("Trading Strategy Performance")
plt.grid(True)
plt.show()