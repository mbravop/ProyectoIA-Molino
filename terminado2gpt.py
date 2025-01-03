#ESTE CODIGO SOLO PREDICE EL SIGUIENTE VALOR. SECUENCIAS DE 24 HORAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, Dropout
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
from keras.src.saving import load_model
from sklearn.metrics import mean_absolute_error, r2_score

# Cargar datos preprocesados
df = pd.read_csv('datosPreProcesados.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

# Crear características de energía y caudal
df['lag_24'] = df['Energía'].shift(24)
df['lag_168'] = df['Energía'].shift(168)
df['rolling_mean_24'] = df['Energía'].rolling(window=24).mean()
df['rolling_mean_168'] = df['Energía'].rolling(window=168).mean()

df['caudal_lag_1'] = df['Caudal'].shift(1)
df['caudal_lag_24'] = df['Caudal'].shift(24)
df['caudal_rolling_mean_24'] = df['Caudal'].rolling(window=24).mean()
df['caudal_rolling_mean_168'] = df['Caudal'].rolling(window=168).mean()

# Eliminar valores NaN generados por los lags y promedios
df.dropna(inplace=True)

# Escalado de los datos (incluyendo caudal)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Energía', 'Caudal', 'lag_24', 'lag_168', 'rolling_mean_24', 'rolling_mean_168',
                                       'caudal_lag_1', 'caudal_lag_24', 'caudal_rolling_mean_24', 'caudal_rolling_mean_168']])
df_scaled = pd.DataFrame(scaled_data, columns=['Energía', 'Caudal', 'lag_24', 'lag_168', 'rolling_mean_24', 'rolling_mean_168',
                                               'caudal_lag_1', 'caudal_lag_24', 'caudal_rolling_mean_24', 'caudal_rolling_mean_168'],
                         index=df.index)

# Crear secuencias para LSTM
def create_sequences(data, target_col, lookback=24):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])  # Secuencia de entradas
        y.append(data[i+lookback, target_col])  # Valor objetivo
    return np.array(X), np.array(y)

lookback = 24
X, y = create_sequences(scaled_data, target_col=0, lookback=lookback)  # Target: energía

# División en conjuntos de entrenamiento y prueba
train_size = len(df.loc['2019':'2023'])
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Crear el modelo LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(lookback, X.shape[2])),  # X.shape[2] incluye todas las características
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

# Configurar callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('mejor_modelo2.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Compilar y entrenar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenamiento

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    callbacks=[early_stopping, checkpoint],
    batch_size=32,
    verbose=1
)

# Cargar el mejor modelo guardado
mejor_modelo = load_model('mejor_modelo2.keras')

# Predicciones
y_pred = mejor_modelo.predict(X_test)

# Desescalar los valores
y_test_rescaled = scaler.inverse_transform([[y, 0, 0, 0, 0, 0, 0, 0, 0, 0] for y in y_test])[:, 0]
y_pred_rescaled = scaler.inverse_transform([[y, 0, 0, 0, 0, 0, 0, 0, 0, 0] for y in y_pred.flatten()])[:, 0]

# Métricas de evaluación
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
r2 = r2_score(y_test_rescaled, y_pred_rescaled)
print(f"Error Absoluto Medio (MAE): {mae}")
print(f"Coeficiente de Determinación (R²): {r2}")

# Visualización de resultados
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label='Valores Reales', alpha=0.7)
plt.plot(y_pred_rescaled, label='Predicciones', alpha=0.7)
plt.title('Comparación de Valores Reales vs. Predicciones')
plt.xlabel('Índice')
plt.ylabel('Energía')
plt.legend()
plt.show()

# Gráfico de dispersión
plt.figure(figsize=(6, 6))
plt.scatter(y_test_rescaled, y_pred_rescaled, alpha=0.5)
plt.plot([y_test_rescaled.min(), y_test_rescaled.max()], [y_test_rescaled.min(), y_test_rescaled.max()], 'r--')
plt.title('Gráfico de Dispersión: Predicciones vs. Valores Reales')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.show()

# Evaluación del mejor modelo guardado
loss = mejor_modelo.evaluate(X_test, y_test, verbose=1)
print(f'Pérdida en el conjunto de prueba: {loss}')

# Calcular el MAPE
non_zero_indices = y_test_rescaled != 0
y_test_filtered = y_test_rescaled[non_zero_indices]
y_pred_filtered = y_pred_rescaled[non_zero_indices]
mape = np.mean(np.abs((y_test_filtered - y_pred_filtered) / y_test_filtered)) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
