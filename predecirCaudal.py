import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, Dropout
from keras.src.callbacks import EarlyStopping
from keras.src.regularizers import L2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Cargar los datos
data = pd.read_csv('datosCompletosMOL.csv')  # Asegúrate de cambiar 'data.csv' por el nombre correcto de tu archivo

# Convertir la columna TimeStamp a formato datetime, si aplica
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data.set_index('Timestamp', inplace=True)

# Escalar los datos
data_values = data['Caudal'].values.reshape(-1, 1)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_values)

# Crear ventanas de tiempo
def create_sequences(data, window_size):
    sequences = []
    targets = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i + window_size])
        targets.append(data[i + window_size])
    return np.array(sequences), np.array(targets)

WINDOW_SIZE = 168  # Una semana (168 horas)
X, y = create_sequences(data_scaled, WINDOW_SIZE)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

# Construir el modelo LSTM
model = Sequential([
    LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer=L2(0.001)),
    Dropout(0.4),
    Dense(1, kernel_regularizer=L2(0.001))
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1,
    validation_split=0.1,
)

# Evaluar el modelo
loss = model.evaluate(X_test, y_test)
print(f'Mean Squared Error: {loss}')

# Predicciones
y_pred = model.predict(X_test)
y_pred_inverse = scaler.inverse_transform(y_pred)
y_test_inverse = scaler.inverse_transform(y_test)

# Guardar el modelo
model.save('lstm_caudal_model.keras')

# Graficar resultados
plt.plot(y_test_inverse, label='Real')
plt.plot(y_pred_inverse, label='Predicho')
plt.legend()
plt.show()
