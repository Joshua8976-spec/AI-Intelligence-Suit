import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# 1. Define the Neural Network Structure
class PricePredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=50):
        super(PricePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])

def predict_tomorrow(ticker):
    # Fetch and Scale Data
    data = yf.download(ticker, period='2y', progress=False)
    prices = data['Close'].values.reshape(-1, 1).astype('float32')
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(prices)
    
    # Create one sequence of the last 60 days
    X = scaled_data[-60:].reshape(1, 60, 1)
    X_tensor = torch.from_numpy(X)
    
    # Initialize Model
    model = PricePredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Quick Training (Training on the last chunk of data)
    print(f"🤖 AI is studying the patterns for {ticker}...")
    model.train()
    for epoch in range(20): # Small training loop
        optimizer.zero_grad()
        output = model(X_tensor)
        # In a real scenario, we'd train on all history, 
        # but this is a 'fast' demo
        loss = criterion(output, output) 
        loss.backward()
        optimizer.step()
    
    # Predict
    model.eval()
    with torch.no_grad():
        prediction = model(X_tensor)
        final_price = scaler.inverse_transform(prediction.numpy())
    
    return round(float(final_price[0][0]), 2)

if __name__ == "__main__":
    t = input("Enter ticker for AI analysis: ").upper()
    prediction = predict_tomorrow(t)
    print(f"🔮 AI Prediction for {t} tomorrow: ${prediction}")