# 8-bit-stock-ticker-prediction

PyTorch LSTM-based stock ticker prediction with dynamic int8 quantization, optimized for NVIDIA RTX 3080ti.

**NOTE** This is just a test and does not predict prices very accurately as the market is influnced by many factors outside of these params.

## Features

- **LSTM Model**: 2-layer LSTM with 50 hidden units
- **Training**: 10 epochs using MSELoss and Adam optimizer
- **Dynamic Quantization**: int8 quantization for ~70% model size reduction
- **Time Series Forecasting**: 60-minute lookback, 15-minute prediction horizon
- **Single-file Implementation**: Complete solution in one Python script

## Requirements

- Python 3.8+
- PyTorch 2.0.0+
- NumPy 1.24.0+

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the stock prediction script:

```bash
python3 stock_prediction.py
```

The script will:
1. Load stock data from `ticker.json`
2. Normalize and create sequences (60m lookback, 15m horizon)
3. Train an LSTM model for 10 epochs
4. Apply dynamic int8 quantization
5. Save both models (`lstm_model_original.pth` and `lstm_model_quantized.pth`)
6. Print model sizes to demonstrate compression
7. Run inference with the quantized model to predict the next 15 minutes

## Data Format

The `ticker.json` file should contain an array of objects with a `close` parameter:

```json
[
  {"close": 1000.50},
  {"close": 1001.25},
  {"close": 999.75},
  ...
]
```

## Results

Example output:
- Original model size: 0.1251 MB
- Quantized model size: 0.0379 MB
- Compression: 69.71% reduction

## Model Architecture

- Input: Sequence of 60 minutes of closing prices
- LSTM Layer 1: 50 hidden units
- LSTM Layer 2: 50 hidden units
- Fully Connected: 15 output units (15-minute predictions)

## License

MIT
