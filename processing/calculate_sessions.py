import numpy as np

def calculate_sessions_expected(eeg_data):
    """
    Calcula el número de sesiones esperadas como la media del número de sesiones realizadas para todos los sujetos.
    """
    sessions_per_subject = [len(sessions) for sessions in eeg_data.values()]  # Contar sesiones para cada sujeto
    sessions_expected = np.mean(sessions_per_subject)  # Media de sesiones esperadas para todos los sujetos
    return sessions_per_subject, sessions_expected


def calculate_session_scores(sessions_realized, sessions_expected):
    """
    Calcula el score de completitud de sesiones para cada sujeto usando la fórmula:
    Score = (1 - (|Sessions Realized - Expected Sessions| / Expected Sessions)^2) * 100

    Si el número de sesiones realizadas es igual al esperado, el resultado será 100.
    """
    scores = [
        (1 - (abs(sessions - sessions_expected) / sessions_expected) ** 2) * 100
        for sessions in sessions_realized
    ]
    return scores



def calculate_final_score(scores):
    """
    Calcula el score global del dataset como el promedio de los scores individuales.
    """
    final_score = np.mean(scores)
    return final_score
