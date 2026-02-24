import pandas as pd

# Define your initial "mock" holdings
# 'Ticker', 'Shares', 'Buy_Price'
data = {
    'Ticker': ['AAPL', 'TSLA', 'BTC-USD', 'NVDA'],
    'Shares': [10, 5, 0.1, 15],
    'Buy_Price': [150.00, 200.00, 30000.00, 100.00]
}

df = pd.DataFrame(data)
df.to_csv('portfolio.csv', index=False)
print("✅ portfolio.csv created with your initial holdings!")