from sklearn.metrics import mutual_info_score
import numpy as np

def calculate_mutual_information(features, eeg_labels):
    """
    Procesa la Información Mutua (MI) entre las características y las etiquetas EEG.

    Parámetros:
        eeg_labels (dict): Diccionario con las etiquetas correspondientes de los datos {sujeto: {sesión: etiquetas}}.
        features (dict): Características extraídas previamente, si ya se extrajeron.

    Retorna:
        mi_scores_table (list): Lista de listas que contiene los puntajes MI por característica y sujeto.
                                Cada lista interna tiene la forma [sujeto, score_feature_0, score_feature_1, ...].
        overall_mi_score (float): Puntaje de MI general del dataset.
        mi_feature_avg (list): Promedio de MI para cada característica.
    """

   # Llamar a la función que calcula MI por sujeto y obtener los puntajes
    mi_scores_per_subject = {}
    for subject, sessions in features.items():
        all_features = []
        all_labels = []

        # Concatenar las características y etiquetas de todas las sesiones para cada sujeto
        for session, feature_data in sessions.items():

            etiqueta = eeg_labels[subject][session]  # Obtener la etiqueta para la sesión actual

            # Reducir la dimensión de los canales tomando la media para cada feature
            feature_data_mean = np.mean(feature_data, axis=0)  # Dimensión resultante: (n_ventanas, n_features)

            # Añadir las características y repetir la etiqueta por el número de ventanas
            all_features.append(feature_data_mean)
            all_labels.extend([etiqueta] * feature_data_mean.shape[0])

        all_features = np.concatenate(all_features, axis=0)  # Concatenar por el eje de ventanas
        all_labels = np.array(all_labels)
        # Asegurarse de que las etiquetas sean 1D
        all_labels = all_labels.ravel()

        # Calcular la Información Mutua para cada característica
        subject_mi_scores = []
        for feature_index in range(all_features.shape[1]):
            feature_data = all_features[:, feature_index]
            mi_score = mutual_info_score(all_labels, feature_data)
            subject_mi_scores.append(mi_score)

        mi_scores_per_subject[subject] = subject_mi_scores

    # Crear la tabla de puntajes de MI para cada característica
    mi_scores_table = []
    for subject, scores in mi_scores_per_subject.items():
        mi_scores_table.append([subject] + scores)

    # Calcular el puntaje general de MI
    overall_mi_score = np.mean([np.mean(scores) for scores in mi_scores_per_subject.values()])

    # Convertir el puntaje general de MI a un rango de 0 a 100
    overall_mi_score = round((np.mean([np.mean(scores) for scores in mi_scores_per_subject.values()]) / 1.0) * 100, 2) if overall_mi_score > 0 else 0

    # Calcular el promedio de MI para cada característica
    mi_feature_avg = [
        np.mean([subject_scores[feature_index] for subject_scores in mi_scores_per_subject.values()])
        for feature_index in range(len(next(iter(mi_scores_per_subject.values()))))
    ]

    return mi_scores_table, overall_mi_score, mi_feature_avg
