import numpy as np
import pandas as pd

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




def calculate_outlier_score(num_outlier_windows, total_windows):
    """
    Calcula el score basado en la proporción de ventanas normales y con outliers.

    Parámetros:
        num_outlier_windows: Diccionario con el número de ventanas con valores atípicos.
        total_windows: Diccionario con el número total de ventanas.

    Retorna:
        scores: Diccionario con el puntaje de outliers para cada sujeto y sesión.
    """
    scores = {}

    # Iterar sobre los sujetos y sesiones
    for subject_id in num_outlier_windows.keys():
        subject_scores = {}

        for session_id in num_outlier_windows[subject_id].keys():
            session_num_outliers = num_outlier_windows[subject_id][session_id]
            session_total_windows = total_windows[subject_id][session_id]

            session_scores = []
            for num_outliers, total in zip(session_num_outliers, session_total_windows):
                if total == 0:
                    session_scores.append(0)  # Si no hay ventanas, devolver 0 como puntaje
                else:
                    proportion_normals = (total - num_outliers) / total
                    proportion_outliers = num_outliers / total
                    score = (proportion_normals - proportion_outliers) * 100
                    session_scores.append(score)

            subject_scores[session_id] = session_scores

        scores[subject_id] = subject_scores

    return scores

