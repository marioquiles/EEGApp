import numpy as np

def calculate_length_scores(eeg_data):
    """
    Calcula el puntaje relacionado con la longitud de las sesiones para cada canal y el puntaje general.

    Score de canal = (1 - Relative Variance) * 100
    Relative Variance = Variance of Length / Mean of Length

    Parámetros:
        eeg_data: Diccionario con los datos EEG.

    Retorna:
        scores_per_channel: Lista de puntajes para cada canal
        overall_score: Puntaje general del dataset
    """
    # Inicializar listas para almacenar las longitudes por canal
    session_lengths_per_channel = {}

    # Recorrer los datos para calcular la longitud de cada sesión por sujeto y canal
    for subject_id, sessions in eeg_data.items():
        for session_id, session_data in sessions.items():
            for channel_index, channel_data in enumerate(session_data):
                channel_length = len(channel_data)

                if channel_index not in session_lengths_per_channel:
                    session_lengths_per_channel[channel_index] = []

                session_lengths_per_channel[channel_index].append(channel_length)

    # Calcular el promedio y la varianza de la longitud para cada canal
    mean_lengths = {}
    variance_lengths = {}

    for channel_index, lengths in session_lengths_per_channel.items():
        mean_lengths[channel_index] = np.mean(lengths)
        variance_lengths[channel_index] = np.var(lengths)

    # Calcular el puntaje relativo de la varianza para cada canal
    scores_per_channel = []
    for channel_index in session_lengths_per_channel.keys():
        if mean_lengths[channel_index] != 0:
            relative_variance = variance_lengths[channel_index] / mean_lengths[channel_index]
            score = (1 - relative_variance) * 100
        else:
            score = 0
        scores_per_channel.append(score)

    # Puntaje general del dataset
    overall_score = np.mean(scores_per_channel)

    return scores_per_channel, overall_score
