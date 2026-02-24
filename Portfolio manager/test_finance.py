import yfinance as yf

# Choose a ticker (e.g., AAPL for Apple, BTC-USD for Bitcoin, TSLA for Tesla)
ticker = input("Enter a Stock Ticker: ")
data = yf.Ticker(ticker)

# Get the current price
current_price = data.history(period='1d')['Close'].iloc[-1]

print(f"The current price of {ticker} is: ${current_price:.2f}")