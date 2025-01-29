import numpy as np
from itertools import combinations
from collections import Counter

def bhattacharyya_coefficient(mean_a, std_a, mean_b, std_b):
    """Calcula el coeficiente de Bhattacharyya entre dos distribuciones Gaussianas."""
    return 0.25 * np.log(0.25 * ((std_a**2 / std_b**2) + (std_b**2 / std_a**2) + 2)) + \
           0.25 * (((mean_a - mean_b)**2) / (std_a**2 + std_b**2))

def calculate_class_overlap(features, labels):
    overlap_scores_per_subject = {}

    for subject, sessions in features.items():
        print(f"Procesando sujeto: {subject}")
        all_features = []
        all_labels = []

        # Combinar características y etiquetas
        for session, feature_data in sessions.items():
            num_ventanas = feature_data.shape[1]
            etiqueta = labels[subject][session]
            print(f"  Procesando sesión: {session} con shape {feature_data.shape} y etiqueta {etiqueta}")
            all_features.append(feature_data)
            all_labels.extend([etiqueta] * num_ventanas)

        all_features = np.concatenate(all_features, axis=1)  # Shape: (n_canales, n_ventanas, n_features)
        all_labels = np.array(all_labels)

        print(f"  Shape combinado de características: {all_features.shape}")
        print(f"  Total etiquetas combinadas: {all_labels.shape}")

        unique_classes = np.unique(all_labels)
        if len(unique_classes) < 2:
            print(f"Advertencia: No hay suficientes clases para el sujeto {subject}.")
            continue

        # Inicializar almacenamiento
        subject_overlap_scores = {i: [] for i in range(all_features.shape[-1])}

        # Calcular solapamiento para cada par de clases
        for class_a, class_b in combinations(unique_classes, 2):
            print(f"    Comparando clases: {class_a} vs {class_b}")
            mask_class_a = all_labels == class_a
            mask_class_b = all_labels == class_b

            class_a_features = all_features[:, mask_class_a, :]
            class_b_features = all_features[:, mask_class_b, :]

            print(f"    Clase {class_a} shape: {class_a_features.shape}, Clase {class_b} shape: {class_b_features.shape}")

            for feature_index in range(all_features.shape[-1]):
                # Extraer características
                a_feat = class_a_features[:, :, feature_index].reshape(-1)
                b_feat = class_b_features[:, :, feature_index].reshape(-1)

                mean_a, std_a = np.mean(a_feat), np.std(a_feat)
                mean_b, std_b = np.mean(b_feat), np.std(b_feat)

                coefficient = bhattacharyya_coefficient(mean_a, std_a, mean_b, std_b)
                subject_overlap_scores[feature_index].append(coefficient)

        # Promediar coeficientes
        average_scores = {feature: np.mean(scores) if scores else 0
                          for feature, scores in subject_overlap_scores.items()}
        overlap_scores_per_subject[subject] = average_scores
        print(f"  Puntajes promedio: {average_scores}")

    # Puntaje general
    overall_scores = [np.mean(list(scores.values())) for scores in overlap_scores_per_subject.values()]
    overall_overlap_score = np.mean(overall_scores) if overall_scores else 0
    print(f"Puntaje general: {overall_overlap_score}")

    return overlap_scores_per_subject, overall_overlap_score



def calculate_p300_class_overlap(features, labels):
    overlap_scores_per_subject = {}

    # Iterar sobre cada sujeto
    for subject, sessions in features.items():
        print(f"\nProcesando sujeto: {subject}")
        all_features = []
        all_labels = []

        # Iterar sobre cada sesión del sujeto
        for session, feature_data in sessions.items():
            print(f"\n  Procesando sesión: {session}")
            print(f"  Dimensiones de características (feature_data): {feature_data.shape}")
            session_labels = labels[subject][session]  # Obtener etiquetas para la sesión actual
            print(f"  Número de etiquetas disponibles: {len(session_labels)}")
            num_windows = feature_data.shape[1]
            print(f"  Número de ventanas en características: {num_windows}")

            # Verifica que el número de etiquetas coincida con el número de ventanas
            if len(session_labels) != num_windows:
                print(f"  Advertencia: El número de etiquetas no coincide con el número de ventanas para {subject}, {session}.")
                continue  # Saltar esta sesión si hay un desajuste

            # Añadir características y etiquetas
            all_features.append(feature_data)
            all_labels.extend(session_labels)

        # Verificar si hay datos acumulados
        if not all_features or not all_labels:
            print(f"  Advertencia: No hay datos válidos acumulados para el sujeto {subject}.")
            continue

        # Convertir a arrays de numpy
        all_features = np.concatenate(all_features, axis=1)  # Concatenar por el eje de ventanas
        all_labels = np.array(all_labels)
        print(f"\n  Dimensiones después de concatenar características: {all_features.shape}")
        print(f"  Número total de etiquetas acumuladas: {len(all_labels)}")

        # Verificar que el tamaño de all_labels coincida con el eje de ventanas en all_features
        if all_features.shape[1] != len(all_labels):
            print(f"  Error: El número de ventanas en las características no coincide con el número de etiquetas para el sujeto {subject}.")
            continue

        # Proceder con el cálculo si las dimensiones son consistentes
        unique_classes = np.unique(all_labels)
        print(f"  Clases únicas detectadas: {unique_classes}")
        if len(unique_classes) < 2:
            print(f"  Advertencia: No hay suficientes clases para el sujeto {subject}.")
            continue

        # Obtener pares de clases
        class_pairs = list(combinations(unique_classes, 2))
        subject_overlap_scores = {feature: [] for feature in range(all_features.shape[-1])}

        for (class_a, class_b) in class_pairs:
            print(f"    Comparando clases: {class_a} vs {class_b}")
            mask_class_a = all_labels == class_a
            mask_class_b = all_labels == class_b

            class_a_features = all_features[:, mask_class_a, :]
            class_b_features = all_features[:, mask_class_b, :]

            print(f"    Dimensiones de características clase {class_a}: {class_a_features.shape}")
            print(f"    Dimensiones de características clase {class_b}: {class_b_features.shape}")

            if class_a_features.size == 0 or class_b_features.size == 0:
                print(f"    Advertencia: Una de las clases no tiene suficientes datos.")
                continue

            # Calcular para cada característica
            for feature_index in range(class_a_features.shape[-1]):
                mean_a = np.mean(class_a_features[:, :, feature_index])
                std_a = np.std(class_a_features[:, :, feature_index])
                mean_b = np.mean(class_b_features[:, :, feature_index])
                std_b = np.std(class_b_features[:, :, feature_index])

                coefficient = bhattacharyya_coefficient(mean_a, std_a, mean_b, std_b)
                subject_overlap_scores[feature_index].append(coefficient)

        average_subject_overlap_scores = {feature: np.mean(scores) if scores else 0
                                          for feature, scores in subject_overlap_scores.items()}
        overlap_scores_per_subject[subject] = average_subject_overlap_scores
        print(f"  Puntajes promedio de solapamiento para el sujeto {subject}: {average_subject_overlap_scores}")

    all_subject_scores = [np.mean(list(subject_scores.values())) for subject_scores in overlap_scores_per_subject.values()]
    overall_overlap_score = np.mean(all_subject_scores) if all_subject_scores else 0
    print(f"\nPuntaje general de solapamiento: {overall_overlap_score}")

    return overlap_scores_per_subject, overall_overlap_score
