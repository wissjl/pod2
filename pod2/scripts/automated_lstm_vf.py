import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def prepare_data(volatility_series, n_lookback=60):
    """Prepares data for volatility prediction"""
    data = volatility_series.values.reshape(-1, 1)
    data = data[~np.isnan(data).any(axis=1)]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(n_lookback, len(scaled_data)):
        X.append(scaled_data[i - n_lookback:i])
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

def run_lstm_model(input_file, output_base_path):
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

    df = pd.read_csv(input_file, delimiter=';')
    df['timestamp'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M', dayfirst=True)
    df.set_index('timestamp', inplace=True)
    df = df[['BTC']].sort_index()
    df['BTC'] = pd.to_numeric(df['BTC'], errors='coerce')
    df_returns = np.log(df / df.shift(1)).rename(columns={'BTC': 'BTC_returns'}).dropna()
    df_returns['realized_vol'] = df_returns['BTC_returns'].rolling(20).std().dropna()

    n_lookback = 60
    X, y, scaler = prepare_data(df_returns['realized_vol'], n_lookback)

    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    layer_sizes = [2**i for i in range(1, 7)]
    results = []

    for u1 in layer_sizes:
        for u2 in layer_sizes:
            try:
                print(f"Training LSTM model with layers ({u1}, {u2})")
                model = build_lstm_model(u1, u2, (n_lookback, 1))
                model.fit(X_train, y_train, epochs=50, batch_size=32,
                          validation_split=0.2, verbose=0)

                test_pred = model.predict(X_test)
                test_pred = scaler.inverse_transform(test_pred).flatten()

                test_dates = df_returns['realized_vol'].index[-len(test_pred):]
                actual_vol = df_returns['realized_vol'].reindex(test_dates)

                mse = mean_squared_error(actual_vol, test_pred)
                mae = mean_absolute_error(actual_vol, test_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(actual_vol, test_pred)

                results.append({
                    'LSTM1': u1,
                    'LSTM2': u2,
                    'R2': r2,
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae
                })

            except Exception as e:
                print(f"⚠️ Failed for ({u1}, {u2}): {str(e)}")
                results.append({
                    'LSTM1': u1, 'LSTM2': u2,
                    'R2': np.nan, 'MSE': np.nan,
                    'RMSE': np.nan, 'MAE': np.nan
                })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_base_path + ".csv", index=False)

    metrics = ['R2', 'MSE', 'RMSE', 'MAE']
    titles = ['R² Score', 'Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error']

    fig = plt.figure(figsize=(20, 16))
    for i, (metric, title) in enumerate(zip(metrics, titles), 1):
        ax = fig.add_subplot(2, 2, i, projection='3d')
        pivot_df = results_df.pivot(index='LSTM1', columns='LSTM2', values=metric)
        X_, Y_ = np.meshgrid(pivot_df.columns, pivot_df.index)
        Z = pivot_df.values
        surf = ax.plot_surface(X_, Y_, Z, cmap='viridis', edgecolor='k', alpha=0.8)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        ax.set_xlabel('LSTM2 Units')
        ax.set_ylabel('LSTM1 Units')
        ax.set_zlabel(title)
        ax.set_title(f'LSTM Performance: {title}')
        ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.savefig(output_base_path + ".png")
    plt.close()
