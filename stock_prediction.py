#!/usr/bin/env python3
"""
LSTM Stock Ticker Prediction with Dynamic Quantization
Optimized for NVIDIA RTX using PyTorch
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
    """Load ticker data from JSON and normalize features (open, high, low, close, vwap).
    Returns normalized 2D array (T, features) and the min/max for the close price for denormalization.
    """
    # Support running from different working directories by resolving relative paths
    tried_paths = [file_path]
    if not os.path.isabs(file_path) and not os.path.exists(file_path):
        # Try relative to this script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt = os.path.normpath(os.path.join(script_dir, file_path))
        tried_paths.append(alt)
        file_path = alt

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found. Tried: {tried_paths}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in '{file_path}': {e}")
    
    # Feature order we will use
    feature_names = ['open', 'high', 'low', 'close', 'vwap']

    # Build a 2D array (T, features)
    try:
        arr = np.array([[float(item.get(k, 0.0)) for k in feature_names] for item in data], dtype=np.float32)
    except Exception as e:
        raise ValueError(f"Error extracting features from JSON: {e}")

    # Compute per-feature min/max for normalization
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    ranges = maxs - mins

    # Avoid division by zero for constant columns
    ranges[ranges < 1e-8] = 1.0

    normalized = (arr - mins) / ranges

    # Return normalized data and close min/max for denormalization
    close_idx = feature_names.index('close')
    min_close = mins[close_idx]
    max_close = maxs[close_idx]

    return normalized, min_close, max_close


def create_sequences(data, lookback=60, horizon=15, feature_index_close=3):
    """Create sequences for multivariate data.

    data: np.ndarray shape (T, features)
    Returns X shape (N, lookback, features) and y shape (N, horizon) containing
    future close values expressed as relative returns from the last close in the
    lookback window: (future_close - last_close) / last_close.
    Predicting returns instead of raw prices improves stability across scales.
    """
    X, y = [], []
    total = len(data)
    for i in range(total - lookback - horizon + 1):
        seq = data[i:i + lookback]
        X.append(seq)
        last_close = seq[-1, feature_index_close]
        future_closes = data[i + lookback:i + lookback + horizon, feature_index_close]
        # relative returns; small epsilon to avoid div-by-zero
        rel = (future_closes - last_close) / (last_close + 1e-8)
        y.append(rel)

    return np.array(X), np.array(y)


# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=2, output_size=15, dropout=0.0, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM dropout is applied between layers when num_layers > 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        fc_in = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_in, output_size)
    
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        if self.bidirectional:
            # adjust hidden/cell for bidirectional (num_layers * num_directions, ...)
            h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        out = self.fc(out[:, -1, :])
        return out

def train_model(model, X_train, y_train, epochs=50, lr=1e-3, batch_size=64, device=None, use_amp=False, val_split=0.1, weight_decay=1e-5, grad_clip=1.0, early_stopping_patience=6):
    """Train the LSTM model using minibatches with validation, scheduler, and early stopping.

    Targets are expected to be relative returns (shape N x horizon).
    Returns the trained model moved to CPU.
    """
    # Huber loss is more robust to outliers than MSE for financial series
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Prepare tensors (stay on CPU until batch is sent)
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    # Simple validation split
    num_samples = len(X_tensor)
    val_count = max(1, int(num_samples * val_split))
    train_count = num_samples - val_count

    train_dataset = torch.utils.data.TensorDataset(X_tensor[:train_count], y_tensor[:train_count])
    val_dataset = torch.utils.data.TensorDataset(X_tensor[train_count:], y_tensor[train_count:])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Choose device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    model = model.to(device)
    print(f"Training on device: {device}")
    print(f"Batch size: {batch_size} | Epochs: {epochs} | Train samples: {train_count} | Val samples: {val_count}")

    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == 'cuda') else None

    # LR scheduler that reduces LR on plateau of validation loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    model.train()
    best_val = float('inf')
    patience = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(xb)
                    loss = criterion(outputs, yb)
                scaler.scale(loss).backward()
                # gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_dataset)

        # validation
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                outputs = model(xb)
                vloss = criterion(outputs, yb)
                val_running += vloss.item() * xb.size(0)
        val_loss = val_running / max(1, len(val_dataset))

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # scheduler step
        scheduler.step(val_loss)

        # early stopping
        if val_loss < best_val - 1e-9:
            best_val = val_loss
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= early_stopping_patience:
                print(f"Early stopping after {epoch+1} epochs. Best val loss: {best_val:.6f}")
                # load best state
                model.load_state_dict(best_state)
                model.to('cpu')
                return model

        model.train()

    # Move back to CPU before quantization/saving
    model.load_state_dict(best_state)
    return model.to('cpu')

def get_model_size(model_path):
    """Get model file size in MB."""
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


def run_inference(model, last_sequence, min_price, max_price, feature_index_close=3):
    """Run inference to predict next `horizon` minutes.

    The model is expected to predict relative returns; convert them to absolute
    prices using the last close in `last_sequence` (which is normalized).
    """
    model.eval()
    with torch.no_grad():
        # Prepare input
        # last_sequence expected shape (lookback, features)
        input_tensor = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0)

        # Predict relative returns
        prediction = model(input_tensor)
        prediction_np = prediction.squeeze().cpu().numpy()

        # Denormalize last close
        last_close_norm = last_sequence[-1, feature_index_close]
        last_close = last_close_norm * (max_price - min_price) + min_price

        # Convert relative returns to prices
        predicted_prices = last_close * (1.0 + prediction_np)

        return predicted_prices

def main():
    print("=" * 60)
    print("LSTM Stock Ticker Prediction with Dynamic Quantization")
    print("Optimized for NVIDIA RTX 3080ti")
    print("=" * 60)
    
    # Load and preprocess data
    print("\n[1] Loading and normalizing data from ticker json...")
    normalized_data, min_price, max_price = load_and_normalize_data()
    print(f"Loaded {len(normalized_data)} data points with {normalized_data.shape[1]} features")
    print(f"Close price range: ${min_price:.2f} - ${max_price:.2f}")
    
    # Ensure enough data for lookback + horizon
    lookback = 60
    horizon = 15
    required = lookback + horizon
    if len(normalized_data) < required + 1:
        raise ValueError(f"Not enough data. Need at least {required+1} points (lookback+ horizon + 1).")
    
    # Reserve last `horizon` rows as held-out (skip them when training)
    held_out = normalized_data[-horizon:, :]
    training_data = normalized_data[:-horizon, :]
    print(f"\nReserved last {horizon} tickers for validation (skipped during training).")
    print(f"Training data length: {len(training_data)}, Held-out length: {len(held_out)}")
    
    # Create sequences from training data
    print("\n[2] Creating sequences (60m lookback, 15m horizon) from training data...")
    # feature_index_close is 3 given our feature ordering in load_and_normalize_data
    feature_index_close = 3
    X, y = create_sequences(training_data, lookback=lookback, horizon=horizon, feature_index_close=feature_index_close)
    print(f"Created {len(X)} sequences")
    print(f"Input shape: {X.shape}, Output shape: {y.shape}")
    
    # Initialize model
    print("\n[3] Initializing LSTM model (hidden=50, layers=2)...")
    # Enable small dropout and bidirectionality as a regularization/expressiveness boost
    model = LSTMModel(input_size=normalized_data.shape[1], hidden_size=50, num_layers=2, output_size=horizon, dropout=0.2, bidirectional=True)
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
    
    # Run inference with quantized model to predict the held-out horizon
    print("\n[8] Running inference with quantized model to predict the held-out 15 tickers...")
    # Use the 60 points immediately before the held-out segment (multivariate)
    last_sequence_start = -lookback - horizon
    last_sequence_end = -horizon if -horizon != 0 else None
    last_sequence = normalized_data[last_sequence_start:last_sequence_end, :]
    if last_sequence.shape[0] != lookback:
        raise ValueError(f"Expected last_sequence of length {lookback}, got {last_sequence.shape[0]}")
    
    predictions = run_inference(quantized_model, last_sequence, min_price, max_price)
    
    # Denormalize actual held-out close values for comparison
    # held_out shape is (horizon, features); close is at index feature_index_close (3)
    actuals_denorm = (held_out[:, feature_index_close] * (max_price - min_price)) + min_price

    print("\nPredicted vs Actual prices for the held-out 15 minutes:")
    print(f"{'Minute':>6}  {'Predicted':>12}  {'Actual':>12}  {'Accuracy':>9}")
    for i, (pred, actual) in enumerate(zip(predictions, actuals_denorm), 1):
        p = float(pred)
        a = float(actual)
        # If actual price is zero, avoid division-by-zero. Treat perfect-zero match as 100% accurate.
        if a == 0.0:
            accuracy = 100.0 if p == 0.0 else 0.0
        else:
            abs_pct = abs(p - a) / abs(a)
            # Define accuracy as (1 - abs_pct) * 100, clamped to [0, 100]
            accuracy = max(0.0, min(100.0, (1.0 - abs_pct) * 100.0))

        print(f"  {i:2d}      ${p:10.2f}    ${a:10.2f}    {accuracy:7.2f}%")
    
    print("\n" + "=" * 60)
    print("COMPLETE! Both models saved and held-out predictions printed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
