import numpy as np
from itertools import combinations

def bhattacharyya_coefficient(mean_a, std_a, mean_b, std_b):
    """Calcula el coeficiente de Bhattacharyya entre dos distribuciones Gaussianas."""
    return 0.25 * np.log(0.25 * ((std_a**2 / std_b**2) + (std_b**2 / std_a**2) + 2)) + \
           0.25 * (((mean_a - mean_b)**2) / (std_a**2 + std_b**2))

def calculate_class_overlap(features, labels):
    overlap_scores_per_subject = {}

    # Iterar sobre cada sujeto
    for subject, sessions in features.items():
        # Inicializar listas para almacenar las características y etiquetas de todas las ventanas del sujeto
        all_features = []
        all_labels = []

        # Iterar sobre cada sesión del sujeto
        for session, feature_data in sessions.items():
            num_ventanas = feature_data.shape[1]
            etiqueta = labels[subject][session]  # Obtener la etiqueta para la sesión actual

            # Añadir las características y repetir la etiqueta por el número de ventanas
            all_features.append(feature_data)
            all_labels.extend([etiqueta] * num_ventanas)

        # Convertir a arrays de numpy para facilitar el procesamiento
        all_features = np.concatenate(all_features, axis=1)  # Concatenar por el eje de ventanas
        all_labels = np.array(all_labels)

        # Debugging prints
        print(f"Procesando el sujeto: {subject}")
        print(f"  Tamaño de all_features: {all_features.shape}")
        print(f"  Tamaño de all_labels: {all_labels.shape}")
        print(f"  all_labels: {all_labels}")

        # Obtener las clases únicas y comprobar si hay al menos dos clases
        unique_classes = np.unique(all_labels)
        print(f"  Etiquetas únicas: {unique_classes}")

        if len(unique_classes) < 2:
            print(f"  Advertencia: No hay suficientes clases para calcular el solapamiento.")
            continue

        # Obtener todas las combinaciones posibles de pares de clases
        class_pairs = list(combinations(unique_classes, 2))
        print(f"  Pares de clases: {class_pairs}")

        # Inicializar diccionario para almacenar los coeficientes para cada característica
        subject_overlap_scores = {feature: [] for feature in range(all_features.shape[-1])}

        # Calcular el coeficiente de Bhattacharyya para cada par de clases y cada característica
        for (class_a, class_b) in class_pairs:
            # Filtrar las ventanas por cada clase
            mask_class_a = all_labels == class_a
            mask_class_b = all_labels == class_b

            class_a_features = all_features[:, mask_class_a, :]
            class_b_features = all_features[:, mask_class_b, :]

            print(f"  Clase A ({class_a}): {class_a_features.shape}")
            print(f"  Clase B ({class_b}): {class_b_features.shape}")

            if class_a_features.size == 0 or class_b_features.size == 0:
                # Si una clase no tiene suficientes ejemplos, no se puede calcular el coeficiente
                continue

            # Calcular la media y desviación estándar para cada característica y clase
            for feature_index in range(class_a_features.shape[-1]):
                # Calcular para clase A
                mean_a = np.mean(class_a_features[:, :, feature_index])
                std_a = np.std(class_a_features[:, :, feature_index])

                # Calcular para clase B
                mean_b = np.mean(class_b_features[:, :, feature_index])
                std_b = np.std(class_b_features[:, :, feature_index])


                # Calcular el coeficiente de Bhattacharyya
                coefficient = bhattacharyya_coefficient(mean_a, std_a, mean_b, std_b)

                subject_overlap_scores[feature_index].append(coefficient)

        # Promediar los coeficientes por característica para el sujeto
        average_subject_overlap_scores = {feature: np.mean(scores) if len(scores) > 0 else 0
                                          for feature, scores in subject_overlap_scores.items()}


        # Guardar los resultados por sujeto
        overlap_scores_per_subject[subject] = average_subject_overlap_scores

    # Calcular el puntaje general de solapamiento para el dataset
    all_subject_scores = [np.mean(list(subject_scores.values())) for subject_scores in overlap_scores_per_subject.values()]
    overall_overlap_score = np.mean(all_subject_scores) if len(all_subject_scores) > 0 else 0


    return overlap_scores_per_subject, overall_overlap_score
