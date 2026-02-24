import pandas as pd
import yfinance as yf

def update_portfolio():
    # 1. Load your holdings
    df = pd.read_csv('portfolio.csv')
    
    # 2. Prepare to store new data
    live_prices = []
    total_values = []
    profits = []

    print("\n📊 FETCHING LIVE DATA...")
    print("-" * 40)

    for index, row in df.iterrows():
        ticker = row['Ticker']
        # Fetch live data
        stock = yf.Ticker(ticker)
        current_price = stock.history(period='1d')['Close'].iloc[-1]
        
        # Calculations
        current_value = current_price * row['Shares']
        cost_basis = row['Buy_Price'] * row['Shares']
        profit = current_value - cost_basis
        
        live_prices.append(round(current_price, 2))
        total_values.append(round(current_value, 2))
        profits.append(round(profit, 2))
        
        print(f"✅ {ticker}: ${current_price:.2f} (P/L: ${profit:+.2f})")

    # 3. Add results to our view
    df['Current_Price'] = live_prices
    df['Total_Value'] = total_values
    df['Profit_Loss'] = profits
    
    print("-" * 40)
    print(f"💰 TOTAL PORTFOLIO VALUE: ${df['Total_Value'].sum():,.2f}")
    print(f"📈 TOTAL PROFIT/LOSS: ${df['Profit_Loss'].sum():+.2f}")
    
    return df

if __name__ == "__main__":
    report = update_portfolio()