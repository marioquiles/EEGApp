import numpy as np
from collections import Counter

def calculate_class_imbalance_score(labels):
    """
    Calcula el puntaje de desbalance de clases por sujeto y la puntuación general.

    Parámetros:
        labels (dict): Diccionario de etiquetas de cada sujeto, con sesiones y sus respectivas etiquetas (puede haber más de dos clases).

    Retorna:
        imbalance_scores (list): Lista de puntajes de desbalance de clases por sujeto.
        total_imbalance_score (float): Puntaje general de desbalance de clases en el rango [0, 100], donde 100 indica balance perfecto.
    """
    imbalance_scores = []

    # Iterar sobre cada sujeto para calcular el puntaje de desbalance de clases
    for subject_id, sessions in labels.items():
        # Concatenar las etiquetas de todas las sesiones para el sujeto actual
        subject_labels = []
        for session_id, label in sessions.items():
            subject_labels.append(label)

        subject_labels = np.array(subject_labels)

        # Contar la cantidad de ejemplos en cada clase para el sujeto actual
        class_counts = Counter(subject_labels)
        counts = np.array(list(class_counts.values()))

        # Evitar división por cero
        if len(counts) <= 1 or np.sum(counts) == 0:
            imbalance_scores.append(0.0)  # Si hay una sola clase o no hay ejemplos, el puntaje de desbalance es 0
        else:
            # Calcular el puntaje de desbalance para el sujeto actual
            min_count = np.min(counts)
            max_count = np.max(counts)
            imbalance_score = (min_count / max_count) * 100  # Puntaje entre 0 y 100
            imbalance_scores.append(round(imbalance_score, 2))

    # Calcular el puntaje general de desbalance de clases como la media de los puntajes por sujeto
    total_imbalance_score = round(np.mean(imbalance_scores), 2)

    return imbalance_scores, total_imbalance_score