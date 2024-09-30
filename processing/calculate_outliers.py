import numpy as np

import numpy as np
import pandas as pd

def calculate_outliers(eeg_data, sampling_frequency, window_size, overlap=0):
    """
    Calcula el número de ventanas con valores atípicos y el total de ventanas.

    Parámetros:
        eeg_data: Array de datos EEG con dimensiones (sujetos, sesiones, canales, muestras).
        sampling_frequency: Frecuencia de muestreo del archivo.
        window_size: Tamaño de la ventana.
        overlap: Porcentaje de solapamiento entre ventanas (valor entre 0 y 1).

    Retorna:
        num_outlier_windows: Número de ventanas con valores atípicos por sujeto y canal.
        total_windows: Número total de ventanas por sujeto y canal.
    """
    if window_size < (sampling_frequency / 4) or window_size > eeg_data.shape[-1]:
        raise ValueError("El tamaño de la ventana no cumple con las restricciones.")
    
    # Calcular el número de pasos de la ventana teniendo en cuenta el solapamiento
    step_size = int(window_size * (1 - overlap))
    if step_size <= 0:
        raise ValueError("El tamaño del paso debe ser mayor que cero. Ajusta el solapamiento o el tamaño de la ventana.")
    
    # Inicializar las listas para almacenar resultados
    num_outlier_windows = []
    total_windows = []

    # Iterar sobre sujetos, sesiones y canales
    for subject in range(eeg_data.shape[0]):
        subject_outlier_windows = []
        subject_total_windows = []
        for session in range(eeg_data.shape[1]):
            for channel in range(eeg_data.shape[2]):
                channel_data = eeg_data[subject, session, channel, :]
                rolling_windows = pd.Series(channel_data).rolling(window=window_size, min_periods=1, step=step_size)

                # Definir magnitud media del canal
                mean_magnitude = np.mean(np.abs(channel_data))
                
                # Contar ventanas con outliers
                num_outliers = 0
                total = 0

                for window in rolling_windows:
                    total += 1
                    if np.any(np.abs(window) > 4 * mean_magnitude):
                        num_outliers += 1

                subject_outlier_windows.append(num_outliers)
                subject_total_windows.append(total)

        num_outlier_windows.append(subject_outlier_windows)
        total_windows.append(subject_total_windows)

    return num_outlier_windows, total_windows


def calculate_outlier_score(num_outlier_windows, total_windows):
    """
    Calcula el score basado en la proporción de ventanas normales y con outliers.

    Parámetros:
        num_outlier_windows: Número de ventanas con valores atípicos.
        total_windows: Número total de ventanas.

    Retorna:
        score: Puntaje en un rango de [0-100].
    """
    if total_windows == 0:
        return 0  # Si no hay ventanas, devolver 0 como puntaje

    proportion_normals = (total_windows - num_outlier_windows) / total_windows
    proportion_outliers = num_outlier_windows / total_windows
    score = (proportion_normals - proportion_outliers) * 100

    return score
