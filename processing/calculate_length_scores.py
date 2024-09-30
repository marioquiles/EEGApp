import numpy as np

def calculate_length_scores(eeg_data):
    """
    Calcula el puntaje relacionado con la longitud de las sesiones para cada canal y el puntaje general.

    Score de canal = (1 - Relative Variance) * 100
    Relative Variance = Variance of Length / Mean of Length

    Parámetros:
        eeg_data: numpy array de dimensiones [num_subjects, num_sessions, num_channels, num_samples]

    Retorna:
        scores_per_channel: Lista de puntajes para cada canal
        overall_score: Puntaje general del dataset
    """
    # Calcular la longitud de cada sesión (en términos de número de muestras)
    num_subjects, num_sessions, num_channels, _ = eeg_data.shape
    session_lengths = np.sum(~np.isnan(eeg_data), axis=3)  # Longitud por [sujeto, sesión, canal]

    # Promedio y varianza de la longitud para cada canal
    mean_lengths = np.mean(session_lengths, axis=(0, 1))  # Promedio de longitud para cada canal
    variance_lengths = np.var(session_lengths, axis=(0, 1))  # Varianza de longitud para cada canal

    # Calcular el puntaje relativo de la varianza para cada canal
    relative_variance = variance_lengths / mean_lengths
    scores_per_channel = (1 - relative_variance) * 100

    # Puntaje general del dataset
    overall_score = np.mean(scores_per_channel)

    return scores_per_channel.tolist(), overall_score
