import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, MaxPooling1D, Flatten, TimeDistributed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def prepare_data_cnn_lstm_vol(df_returns, return_col='ETH_returns', vol_col='Realized_Volatility',
                              n_lookback=60, n_steps=1, test_size=0.2):
    data_x = df_returns[return_col].values.reshape(-1, 1)
    data_y = df_returns[vol_col].values.reshape(-1, 1)
    timestamps = df_returns.index

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaled_x = scaler_x.fit_transform(data_x)
    scaled_y = scaler_y.fit_transform(data_y)

    X, y, time_index = [], [], []
    for i in range(n_lookback, len(scaled_x)):
        X.append(scaled_x[i - n_lookback:i])
        y.append(scaled_y[i])
        time_index.append(timestamps[i])

    X = np.array(X).reshape(-1, n_steps, n_lookback, 1)
    y = np.array(y)
    time_index = np.array(time_index)

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    plot_dates = time_index[split_idx:]

    print(f"Prepared shapes -> X: {X.shape}, y: {y.shape}, Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test, plot_dates, scaler_y

def build_cnn_lstm(u1, u2, input_shape):
    model = Sequential()

    model.add(TimeDistributed(
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(u1, return_sequences=True))
    model.add(LSTM(u2))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    return model

def evaluate_model(y_true, y_pred, scaler):
    y_true = scaler.inverse_transform(y_true.reshape(-1, 1))
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

    metrics = {
        'R2': r2_score(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred)
    }
    return metrics

if __name__ == "__main__":
    LOCAL_INPUT_DIR = os.path.dirname(__file__)
    project_base_dir = os.path.dirname(os.path.dirname(__file__))
    LOCAL_OUTPUT_BASE_DIR = os.path.join(project_base_dir, 'results_local', 'cnn_lstm_outputs')

    os.makedirs(LOCAL_OUTPUT_BASE_DIR, exist_ok=True)

    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("✅ TensorFlow GPU detected and ready on local machine.")
        else:
            print("❌ No GPU found on local machine. Using CPU.")
    except ImportError:
        print("⚠️ TensorFlow not installed — skipping GPU setup.")

    csv_files = [f for f in os.listdir(LOCAL_INPUT_DIR) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in the specified input directory: {LOCAL_INPUT_DIR}")
    else:
        print(f"Found {len(csv_files)} CSV files in {LOCAL_INPUT_DIR}. Starting processing...")

    for filename in csv_files:
        input_file_path = os.path.join(LOCAL_INPUT_DIR, filename)
        
        file_base_name = os.path.splitext(filename)[0]
        current_output_base_path = os.path.join(LOCAL_OUTPUT_BASE_DIR, f"cnn_lstm_{file_base_name}")

        print(f"\n--- Processing input file: {filename} ---")
        print(f"Saving results to: {current_output_base_path}.csv and .png")

        try:
            df = pd.read_csv(input_file_path, delimiter=';')
        except FileNotFoundError:
            print(f"Error: File {input_file_path} not found. Skipping.")
            continue
        except Exception as e:
            print(f"Error reading {input_file_path}: {e}. Skipping.")
            continue

        df['timestamp'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M', dayfirst=True)
        df.set_index('timestamp', inplace=True)
        df = df[['ETH']].sort_index().dropna()

        df_returns = np.log(df / df.shift(1)).dropna()
        df_returns.rename(columns={'ETH': 'ETH_returns'}, inplace=True)
        df_returns['Realized_Volatility'] = df_returns['ETH_returns'].rolling(window=20).std()
        df_returns = df_returns.dropna()

        X_train, X_test, y_train, y_test, plot_dates, scaler_y = prepare_data_cnn_lstm_vol(
            df_returns, n_lookback=60, test_size=0.2)

        layer_sizes = [2**i for i in range(1, 7)]
        results = []

        for u1 in layer_sizes:
            for u2 in layer_sizes:
                try:
                    print(f"\n  Training CNN-LSTM with layers ({u1}, {u2}) for {filename}")
                    model = build_cnn_lstm(u1, u2, X_train.shape[1:])
                    model.fit(X_train, y_train,
                              epochs=30,
                              batch_size=32,
                              validation_data=(X_test, y_test),
                              verbose=1)
                    y_pred = model.predict(X_test, verbose=0)
                    metrics = evaluate_model(y_test, y_pred, scaler_y)
                    results.append({
                        'LSTM1': u1, 'LSTM2': u2,
                        **metrics
                    })
                    print(f"  Results for ({u1}, {u2}): R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}")

                except Exception as e:
                    print(f"⚠️ Failed ({u1}, {u2}) for {filename}: {str(e)}")
                    results.append({'LSTM1': u1, 'LSTM2': u2, 'R2': None, 'MSE': None, 'RMSE': None, 'MAE': None})

        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(current_output_base_path + ".csv", index=False)

            metrics_to_plot = ['R2', 'MSE', 'RMSE', 'MAE']
            titles = ['R² Score', 'Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error']
            colors = ['b', 'r', 'g', 'm']

            fig = plt.figure(figsize=(20, 16))
            for i, (metric, title, color) in enumerate(zip(metrics_to_plot, titles, colors), 1):
                ax = fig.add_subplot(2, 2, i, projection='3d')
                pivot_df = results_df.pivot(index='LSTM1', columns='LSTM2', values=metric)
                X, Y = np.meshgrid(pivot_df.columns, pivot_df.index)
                Z = pivot_df.values
                surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.8)
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
                ax.set_xlabel('LSTM2 Units')
                ax.set_ylabel('LSTM1 Units')
                ax.set_zlabel(title)
                ax.set_title(f'{title}')
                ax.view_init(elev=30, azim=45)
            plt.tight_layout()
            plt.savefig(current_output_base_path + ".png")
            plt.close(fig)
            print(f"Successfully processed {filename} and saved results.")
        else:
            print(f"No results collected for {filename}.")

    print("\n--- All CNN-LSTM processing complete for local files. ---")