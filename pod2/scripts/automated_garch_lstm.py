import os
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from arch import arch_model

def prepare_data(series, n_lookback=60):
    """Prépare les séquences pour le LSTM"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(n_lookback, len(scaled_data)):
        X.append(scaled_data[i-n_lookback:i])
        y.append(scaled_data[i])
    
    return np.array(X), np.array(y), scaler

def build_lstm_model(u1, u2, input_shape):
    model = Sequential([
        LSTM(u1, return_sequences=True, input_shape=input_shape),
        LSTM(u2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def run_garch_lstm_model(input_file, output_base_path):
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("✅ TensorFlow GPU detected and ready.")
        else:
            print("❌ No GPU found. Using CPU.")
    except ImportError:
        print("⚠️ TensorFlow not installed — skipping GPU setup.")

    os.makedirs(os.path.dirname(output_base_path), exist_ok=True)

    # Load and preprocess data
    df = pd.read_csv(input_file, delimiter=';')
    df['timestamp'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M', dayfirst=True)
    df.set_index('timestamp', inplace=True)
    df = df[['BTC']].sort_index()
    df['BTC'] = pd.to_numeric(df['BTC'], errors='coerce')
    df['returns'] = np.log(df['BTC'] / df['BTC'].shift(1))
    df.dropna(inplace=True)

    # GARCH volatility
    df['realized_vol'] = df['returns'].rolling(window=20).std()
    garch = arch_model(df['returns'], vol='Garch', p=1, q=1, dist='normal')
    garch_fit = garch.fit(disp='off')
    df['garch_vol'] = garch_fit.conditional_volatility
    df.dropna(inplace=True)

    # Prepare LSTM data
    n_lookback = 60
    X, y, scaler = prepare_data(df['garch_vol'], n_lookback)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Train with multiple architectures
    layer_sizes = [2**i for i in range(1, 7)]
    results = []

    for u1 in layer_sizes:
        for u2 in layer_sizes:
            try:
                print(f"Training GARCH-LSTM model with layers ({u1}, {u2})")
                model = build_lstm_model(u1, u2, (n_lookback, 1))
                model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
                
                # Prediction
                test_pred = model.predict(X_test)
                test_pred = scaler.inverse_transform(test_pred).flatten()

                # Evaluation
                test_dates = df.index[-len(test_pred):]
                realized_vol_test = df['realized_vol'].reindex(test_dates)

                mse = mean_squared_error(realized_vol_test, test_pred)
                mae = mean_absolute_error(realized_vol_test, test_pred)
                rmse = sqrt(mse)
                r2 = r2_score(realized_vol_test, test_pred)

                results.append({
                    'LSTM1': u1,
                    'LSTM2': u2,
                    'R2': r2,
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae
                })
            except Exception as e:
                print(f"⚠️ Failed ({u1}, {u2}): {str(e)}")
                results.append({
                    'LSTM1': u1, 'LSTM2': u2,
                    'R2': np.nan, 'MSE': np.nan,
                    'RMSE': np.nan, 'MAE': np.nan
                })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_base_path + ".csv", index=False)

    # 3D Plot
    metrics = ['R2', 'MSE', 'RMSE', 'MAE']
    titles = ['R² Score', 'Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error']

    fig = plt.figure(figsize=(20, 16))
    for i, (metric, title) in enumerate(zip(metrics, titles), 1):
        ax = fig.add_subplot(2, 2, i, projection='3d')
        pivot_df = results_df.pivot(index='LSTM1', columns='LSTM2', values=metric)
        X, Y = np.meshgrid(pivot_df.columns, pivot_df.index)
        Z = pivot_df.values
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.8)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        ax.set_xlabel('LSTM Layer 2 Units')
        ax.set_ylabel('LSTM Layer 1 Units')
        ax.set_zlabel(title)
        ax.set_title(f'GARCH-LSTM Performance: {title}')
        ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.savefig(output_base_path + ".png")
    plt.close()
