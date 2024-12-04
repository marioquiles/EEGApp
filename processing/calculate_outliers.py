import numpy as np
import pandas as pd


def calculate_P300outliers(eeg_data, threshold=4):
    """
    Calcula el porcentaje de ventanas con valores atípicos en los datos EEG para P300.

    Parámetros:
        eeg_data: Diccionario con los datos EEG (formato: sujetos -> sesiones -> canales -> ventanas).
        threshold: Z-score para considerar un valor como outlier (por defecto 4).

    Retorna:
        outlier_percentages: Diccionario con el porcentaje de ventanas con outliers por sujeto.
    """
    outlier_percentages = {}

    # Iterar sobre los sujetos
    for subject_id, sessions in eeg_data.items():
        total_windows = 0
        outlier_windows = 0

        # Iterar sobre sesiones y canales para cada sujeto
        for session_data in sessions.values():
            for channel_data in session_data:
                # Calcular la desviación estándar y la media del canal completo para cada ventana de este canal
                mean_channel = np.mean(channel_data)
                std_dev_channel = np.std(channel_data)

                # Iterar sobre cada ventana ya existente
                for window in channel_data:
                    # Comprobar si hay un outlier en la ventana usando la desviación estándar del canal
                    if std_dev_channel > 0:  # Evitar división por cero
                        z_scores = np.abs(window - mean_channel) / std_dev_channel
                        if np.any(z_scores > threshold):
                            outlier_windows += 1

                    total_windows += 1

        # Calcular el porcentaje de ventanas con outliers para el sujeto
        if total_windows > 0:
            percentage_outliers = (outlier_windows / total_windows) * 100
        else:
            percentage_outliers = 0

        outlier_percentages[subject_id] = percentage_outliers

    return outlier_percentages




def calculate_outliers(eeg_data, window_size, overlap=0):
    """
    Calcula el porcentaje de ventanas con valores atípicos en los datos EEG.

    Parámetros:
        eeg_data: Diccionario con los datos EEG (formato: sujetos -> sesiones -> canales).
        window_size: Tamaño de la ventana para dividir los datos.
        overlap: Porcentaje de solapamiento entre ventanas (valor entre 0 y 1).

    Retorna:
        outlier_percentages: Diccionario con el porcentaje de ventanas con outliers por sujeto.
    """
    outlier_percentages = {}

    # Iterar sobre los sujetos
    for subject_id, sessions in eeg_data.items():
        total_windows = 0
        outlier_windows = 0

        # Iterar sobre sesiones y canales para cada sujeto
        for session_data in sessions.values():
            for channel_data in session_data:
                # Calcular la desviación estándar de toda la señal para el canal y sesión actuales
                mean_channel = np.mean(channel_data)
                std_dev_channel = np.std(channel_data)

                # Calcular el número de pasos de la ventana teniendo en cuenta el solapamiento
                step_size = int(window_size - overlap)
                if step_size <= 0:
                    raise ValueError("El tamaño del paso debe ser mayor que cero. Ajusta el solapamiento o el tamaño de la ventana.")

                # Dividir los datos en ventanas
                for i in range(0, len(channel_data) - window_size + 1, step_size):
                    window = channel_data[i:i + window_size]

                    # Comprobar si hay un outlier en la ventana usando la desviación estándar de toda la señal
                    if std_dev_channel > 0:  # Evitar división por cero
                        z_scores = np.abs(window - mean_channel) / std_dev_channel
                        if np.any(z_scores > 4):
                            outlier_windows += 1

                    total_windows += 1

        # Calcular el porcentaje de ventanas con outliers para el sujeto
        if total_windows > 0:
            percentage_outliers = (outlier_windows / total_windows) * 100
        else:
            percentage_outliers = 0

        outlier_percentages[subject_id] = percentage_outliers

    return outlier_percentages