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



def calculate_p300_class_overlap(features, labels, log_file="debug_log.txt"):
    with open(log_file, "w", encoding="utf-8") as log:
        def log_print(message):
            print(message)  # Muestra en pantalla
            log.write(message + "\n")  # Guarda en archivo

        overlap_scores_per_subject = {}

        for subject, sessions in features.items():
            log_print(f"\n🔹 Procesando sujeto: {subject}")
            all_features = []
            all_labels = []

            for session, feature_data in sessions.items():
                log_print(f"\nProcesando sesión: {session}")
                log_print(f"Dimensiones de características: {feature_data.shape}")
                session_labels = labels[subject][session]
                log_print(f" Número de etiquetas disponibles: {len(session_labels)}")
                num_windows = feature_data.shape[1]
                log_print(f" Número de ventanas en características: {num_windows}")

                if len(session_labels) != num_windows:
                    log_print(f"Advertencia: El número de etiquetas no coincide con las ventanas.")
                    continue

                all_features.append(feature_data)
                all_labels.extend(session_labels)

            if not all_features or not all_labels:
                log_print(f"Advertencia: No hay datos válidos para {subject}.")
                continue

            all_features = np.concatenate(all_features, axis=1)
            all_labels = np.array(all_labels)
            log_print(f"\nDimensiones concatenadas: {all_features.shape}")
            log_print(f"Número total de etiquetas: {len(all_labels)}")

            if all_features.shape[1] != len(all_labels):
                log_print(f"Error: Desajuste entre ventanas y etiquetas.")
                continue

            unique_classes = np.unique(all_labels)
            log_print(f"Clases únicas detectadas: {unique_classes}")
            if len(unique_classes) < 2:
                log_print(f"Advertencia: No hay suficientes clases en {subject}.")
                continue

            class_pairs = list(combinations(unique_classes, 2))
            subject_overlap_scores = {feature: [] for feature in range(all_features.shape[-1])}

            for (class_a, class_b) in class_pairs:
                log_print(f"Comparando clases: {class_a} vs {class_b}")
                mask_class_a = all_labels == class_a
                mask_class_b = all_labels == class_b

                class_a_features = all_features[:, mask_class_a, :]
                class_b_features = all_features[:, mask_class_b, :]

                log_print(f"Clase {class_a}: {class_a_features.shape}")
                log_print(f"Clase {class_b}: {class_b_features.shape}")

                if class_a_features.size == 0 or class_b_features.size == 0:
                    log_print(f"Advertencia: Una de las clases no tiene suficientes datos.")
                    continue

                for feature_index in range(class_a_features.shape[-1]):
                    mean_a = np.mean(class_a_features[:, :, feature_index])
                    std_a = np.std(class_a_features[:, :, feature_index])
                    mean_b = np.mean(class_b_features[:, :, feature_index])
                    std_b = np.std(class_b_features[:, :, feature_index])

                    log_print(f"Feature {feature_index}: Mean A = {mean_a:.4f}, Std A = {std_a:.4f}")
                    log_print(f"Feature {feature_index}: Mean B = {mean_b:.4f}, Std B = {std_b:.4f}")

                    if std_a < 1e-6 or std_b < 1e-6:
                        log_print(f"Advertencia: Baja desviación estándar en Feature {feature_index}.")

                    coefficient = bhattacharyya_coefficient(mean_a, std_a, mean_b, std_b)
                    log_print(f"Coeficiente Bhattacharyya (Feature {feature_index}): {coefficient:.4f}")

                    subject_overlap_scores[feature_index].append(coefficient)

            average_subject_overlap_scores = {
                feature: np.mean(scores) if scores else 0 for feature, scores in subject_overlap_scores.items()
            }
            overlap_scores_per_subject[subject] = average_subject_overlap_scores
            log_print(f"Puntajes promedio de solapamiento para {subject}: {average_subject_overlap_scores}")

        all_subject_scores = [np.mean(list(subject_scores.values())) for subject_scores in overlap_scores_per_subject.values()]
        overall_overlap_score = np.mean(all_subject_scores) if all_subject_scores else 0
        log_print(f"\nPuntaje general de solapamiento: {overall_overlap_score:.4f}")

    return overlap_scores_per_subject, overall_overlap_score
