import numpy as np

def calculate_length_scores(eeg_data):
    """
    Calcula el puntaje relacionado con la longitud de las sesiones para cada sujeto y el puntaje general.

    Score de sujeto = max(0, (1 - Coefficient of Variation)) * 100
    Coefficient of Variation = Standard Deviation of Length / Mean of Length

    Parámetros:
        eeg_data: Diccionario con los datos EEG.

    Retorna:
        scores_per_subject: Lista de puntajes para cada sujeto
        overall_score: Puntaje general del dataset
    """
    # Inicializar un diccionario para almacenar las longitudes de las sesiones por sujeto
    session_lengths_per_subject = {}

    # Recorrer los datos para calcular la longitud de cada sesión por sujeto
    for subject_id, sessions in eeg_data.items():
        session_lengths = []

        for session_id, session_data in sessions.items():
            # La longitud de la sesión es la segunda dimensión del array (número de muestras)
            session_length = session_data.shape[1]
            session_lengths.append(session_length)

        session_lengths_per_subject[subject_id] = session_lengths

    # Calcular el promedio y la desviación estándar de la longitud para cada sujeto
    scores_per_subject = []
    for subject_id, lengths in session_lengths_per_subject.items():
        if len(lengths) > 1:  # Asegurarse de que haya suficientes sesiones para calcular una variación
            mean_length = np.mean(lengths)
            std_length = np.std(lengths)

            if mean_length != 0:
                coefficient_of_variation = std_length / mean_length
                score = max(0, (1 - coefficient_of_variation)) * 100
            else:
                score = 0
        else:
            # Si solo hay una sesión, la variación es 0 y el puntaje es 100
            score = 100

        scores_per_subject.append(score)

    # Puntaje general del dataset
    overall_score = np.mean(scores_per_subject)

    return scores_per_subject, overall_score