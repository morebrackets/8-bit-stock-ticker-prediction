#!/usr/bin/env python3
"""
LSTM Stock Ticker Prediction with Dynamic Quantization
Optimized for NVIDIA RTX 3080ti using PyTorch
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.quantization import quantize_dynamic


# Data loading and preprocessing
def load_and_normalize_data(file_path='ticker.json'):
    """Load ticker data from JSON and normalize close prices."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file '{file_path}' not found. Please ensure ticker.json exists.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in '{file_path}': {e}")
    
    # Extract close prices
    close_prices = np.array([item['close'] for item in data], dtype=np.float32)
    
    # Normalize to [0, 1] range
    min_price = close_prices.min()
    max_price = close_prices.max()
    price_range = max_price - min_price
    
    # Handle edge case where all prices are the same
    if price_range < 1e-8:
        normalized_prices = np.zeros_like(close_prices)
    else:
        normalized_prices = (close_prices - min_price) / price_range
    
    return normalized_prices, min_price, max_price


def create_sequences(data, lookback=60, horizon=15):
    """Create sequences with lookback window and prediction horizon."""
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback:i + lookback + horizon])
    
    return np.array(X), np.array(y)


# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=15):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        out = self.fc(out[:, -1, :])
        return out


def train_model(model, X_train, y_train, epochs=10, lr=0.001):
    """Train the LSTM model."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())} total parameters")
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
    
    return model.to('cpu')  # Move back to CPU for quantization


def get_model_size(model_path):
    """Get model file size in MB."""
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


def run_inference(model, last_sequence, min_price, max_price):
    """Run inference to predict next 15 minutes."""
    model.eval()
    with torch.no_grad():
        # Prepare input
        input_tensor = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        
        # Predict
        prediction = model(input_tensor)
        
        # Denormalize predictions
        prediction_np = prediction.squeeze().numpy()
        denormalized = prediction_np * (max_price - min_price) + min_price
        
        return denormalized


def main():
    print("=" * 60)
    print("LSTM Stock Ticker Prediction with Dynamic Quantization")
    print("Optimized for NVIDIA RTX 3080ti")
    print("=" * 60)
    
    # Load and preprocess data
    print("\n[1] Loading and normalizing data from ticker.json...")
    normalized_data, min_price, max_price = load_and_normalize_data()
    print(f"Loaded {len(normalized_data)} data points")
    print(f"Price range: ${min_price:.2f} - ${max_price:.2f}")
    
    # Create sequences
    print("\n[2] Creating sequences (60m lookback, 15m horizon)...")
    X, y = create_sequences(normalized_data, lookback=60, horizon=15)
    print(f"Created {len(X)} sequences")
    print(f"Input shape: {X.shape}, Output shape: {y.shape}")
    
    # Initialize model
    print("\n[3] Initializing LSTM model (hidden=50, layers=2)...")
    model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=15)
    print(f"Model architecture:\n{model}")
    
    # Train model
    print("\n[4] Training model (10 epochs, MSELoss, Adam)...")
    trained_model = train_model(model, X, y, epochs=10)
    
    # Save original model
    print("\n[5] Saving original model...")
    original_model_path = 'lstm_model_original.pth'
    torch.save(trained_model.state_dict(), original_model_path)
    original_size = get_model_size(original_model_path)
    print(f"Original model saved to {original_model_path}")
    print(f"Original model size: {original_size:.4f} MB")
    
    # Apply dynamic quantization
    print("\n[6] Applying dynamic quantization (int8)...")
    trained_model.eval()
    quantized_model = quantize_dynamic(
        trained_model,
        {nn.LSTM, nn.Linear},
        dtype=torch.qint8
    )
    
    # Save quantized model
    print("\n[7] Saving quantized model...")
    quantized_model_path = 'lstm_model_quantized.pth'
    torch.save(quantized_model.state_dict(), quantized_model_path)
    quantized_size = get_model_size(quantized_model_path)
    print(f"Quantized model saved to {quantized_model_path}")
    print(f"Quantized model size: {quantized_size:.4f} MB")
    
    # Calculate compression ratio
    compression_ratio = (1 - quantized_size / original_size) * 100
    print(f"\n{'=' * 60}")
    print(f"COMPRESSION RESULTS:")
    print(f"Original model:  {original_size:.4f} MB")
    print(f"Quantized model: {quantized_size:.4f} MB")
    print(f"Compression:     {compression_ratio:.2f}% reduction")
    print(f"{'=' * 60}")
    
    # Run inference with quantized model
    print("\n[8] Running inference with quantized model...")
    print("Predicting next 15 minutes based on last 60 minutes of data...")
    last_sequence = normalized_data[-60:]
    predictions = run_inference(quantized_model, last_sequence, min_price, max_price)
    
    print("\nPredicted prices for next 15 minutes:")
    for i, price in enumerate(predictions, 1):
        print(f"  Minute {i:2d}: ${price:.2f}")
    
    print("\n" + "=" * 60)
    print("COMPLETE! Both models saved and inference performed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
