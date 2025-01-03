import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def preprocess_hydro_data(df):
    df_clean = df.copy()

    # Función auxiliar para obtener promedio de valores válidos en una ventana
    def get_valid_window_mean(series, index, window_before, window_after):
        start_idx = max(0, index - window_before)
        end_idx = min(len(series), index + window_after + 1)
        window_values = series[start_idx:end_idx]
        valid_values = window_values[window_values != 0]
        return valid_values.mean() if len(valid_values) > 0 else None

    # Procesamiento de Caudal = 0
    zero_flow_mask = df_clean['Caudal'] == 0
    for idx in df_clean[zero_flow_mask].index:
        # Intentar obtener promedio de hora anterior y siguiente
        replacement_value = get_valid_window_mean(df_clean['Caudal'], idx, 1, 1)
        if replacement_value is not None:
            df_clean.loc[idx, 'Caudal'] = replacement_value

    # Procesamiento de energía = 0 con Caudal > 12
    zero_energy_mask = (df_clean['Energía'] == 0) & (df_clean['Caudal'] > 12)

    # Iterar mientras haya valores para reemplazar
    max_iterations = 24  # Límite de iteraciones para evitar bucles infinitos
    iteration = 0

    while zero_energy_mask.any() and iteration < max_iterations:
        for idx in df_clean[zero_energy_mask].index:
            # Intentar obtener promedio de dos horas antes y después
            replacement_value = get_valid_window_mean(df_clean['Energía'], idx, 2, 2)
            if replacement_value is not None:
                df_clean.loc[idx, 'Energía'] = replacement_value

        # Actualizar máscara para próxima iteración
        zero_energy_mask = (df_clean['Energía'] == 0) & (df_clean['Caudal'] > 12)
        iteration += 1

    # Registrar valores que no se pudieron limpiar
    remaining_zeros_flow = df_clean['Caudal'] == 0
    remaining_zeros_energy = (df_clean['Energía'] == 0) & (df_clean['Caudal'] > 12)

    cleaning_stats = {
        'remaining_zero_flow': remaining_zeros_flow.sum(),
        'remaining_zero_energy': remaining_zeros_energy.sum(),
        'iterations_needed': iteration
    }

    return df_clean, cleaning_stats

df = pd.read_csv('datosCompletosMOL.csv')  # Asegúrate de que tenga las columnas correctas
df_limpio, estadisticas = preprocess_hydro_data(df)

print(len(df))
# Ver estadísticas de limpieza
print(estadisticas)
#df_limpio.to_csv('datosPreProcesados.csv', encoding='utf-8', index=False)
print(df['Caudal'].corr(df['Energía']))
print(df_limpio['Caudal'].corr(df_limpio['Energía']))

plt.plot(df_limpio['Caudal'])
plt.show()