from processing.calculate_sessions import calculate_sessions_expected, calculate_session_scores, calculate_final_score
from processing.input_validation import validate_window_size
from processing.calculate_length_scores import calculate_length_scores
from processing.calculate_labels import calculate_class_imbalance_score
from .calculate_outliers import calculate_outliers, calculate_outlier_score
from processing.extract_emotional_features import extract_emotional_features
from processing.calculate_overlap import calculate_class_overlap
from processing.mutual_information import calculate_mutual_information
import numpy as np
import os
import pickle

def process_session_metrics(eeg_data):
    """
    Calcula métricas relacionadas con el número de sesiones por sujeto.

    Parámetros:
        eeg_data: Diccionario con los datos EEG.

    Retorna:
        Un diccionario con los resultados de las métricas relacionadas con sesiones.
    """
    sessions_realized, sessions_expected = calculate_sessions_expected(eeg_data)
    session_scores = calculate_session_scores(sessions_realized, sessions_expected)
    final_score_sessions = calculate_final_score(session_scores)

    results = {
        'session_scores': session_scores,
        'final_score_sessions': final_score_sessions,
        'num_sujetos': len(sessions_realized),
        'sessions_expected': sessions_expected
    }

    return results

def process_length_metrics(eeg_data):
    """
    Calcula métricas relacionadas con la longitud de las sesiones.

    Parámetros:
        eeg_data: Diccionario con los datos EEG.

    Retorna:
        Un diccionario con los resultados de las métricas relacionadas con la longitud de las sesiones.
    """
    # Calcular métricas relacionadas con la longitud de las sesiones
    scores_per_channel, overall_score_length = calculate_length_scores(eeg_data)

    results = {
        'scores_per_channel': scores_per_channel,
        'overall_score_length': overall_score_length
    }

    return results

def process_outlier_metrics(eeg_data, window_size, overlap):
    try:
        # Calcular el porcentaje de ventanas con outliers usando la función calculate_outliers
        outlier_percentages = calculate_outliers(eeg_data, window_size, overlap)

        # Calcular la puntuación final para los outliers
        mean_outlier_percentage = np.mean(list(outlier_percentages.values()))
        final_outlier_score = max(0, 100 - mean_outlier_percentage)
        final_outlier_score = round(final_outlier_score, 2)

        # Preparar el resultado final
        result = {
            'outlier_percentages': outlier_percentages,
            'final_outlier_score': final_outlier_score
        }

        print(result)  # Agregar esto para verificar el contenido del resultado
        return result

    except Exception as e:
        print(f"Error en process_outlier_metrics: {e}")
        return {'error': str(e)}


def process_labels_metrics(eeg_labels):
    """
    Procesa las etiquetas EEG para calcular el desbalance de clases por sujeto y el puntaje total.

    Parámetros:
        eeg_labels (numpy array): Array de dimensiones [num_subjects, num_samples] con las etiquetas de los datos.

    Retorna:
        Un diccionario con el puntaje de desbalance de clases por sujeto y el puntaje total.
    """
    # Calcular el puntaje de desbalance de clases por sujeto y el puntaje total
    imbalance_scores, total_imbalance_score = calculate_class_imbalance_score(eeg_labels)
    
    return {
        'class_imbalance_scores': imbalance_scores,
        'total_imbalance_score': total_imbalance_score
    }

# Función para procesar y guardar características
def process_eeg_features(eeg_data, filename, sampling_rate=128, window_size=256, overlap=128, output_folder="features/"):
    """
    Extrae las características emocionales de los datos EEG y las guarda en un archivo.

    Parámetros:
        eeg_data (dict): Diccionario con los datos EEG de dimensiones {sujeto: {sesión: (num_canales, num_muestras)}}.
        sampling_rate (int): Frecuencia de muestreo de los datos EEG.
        window_size (int): Tamaño de la ventana en muestras.
        overlap (int): Cantidad de solapamiento entre ventanas en muestras.
        output_folder (str): Directorio para guardar los archivos de características.

    Retorna:
        feature_filepath (str): Ruta al archivo donde se han guardado las características.
    """
    # Extraer características usando la función extract_emotional_features
    features = extract_emotional_features(eeg_data, sampling_rate, window_size, overlap)

    # Crear la carpeta de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Guardar las características extraídas en un archivo
    feature_filepath = os.path.join(output_folder, filename + "_features")
    with open(feature_filepath, 'wb') as f:
        pickle.dump(features, f)

    print(f"Características extraídas y guardadas en {feature_filepath}")
    return feature_filepath

def process_class_overlap(eeg_labels, features=None):
    """
    Procesa el solapamiento entre clases usando las características extraídas.

    Parámetros:
        eeg_labels (dict): Diccionario con las etiquetas correspondientes de los datos {sujeto: {sesión: etiquetas}}.
        features (dict): Características extraídas previamente, si ya se extrajeron.

    Retorna:
        overlap_scores_table (list): Lista de listas que contiene los puntajes de solapamiento por característica y sujeto.
                                     Cada lista interna tiene la forma [sujeto, score_feature_0, score_feature_1, ...].
        overall_overlap_score (float): Puntaje de solapamiento general del dataset.
    """

    overlap_scores_per_subject, overall_overlap_score = calculate_class_overlap(features, eeg_labels)

    # Limitar los scores entre 0 y 1
    for subject, scores in overlap_scores_per_subject.items():
        overlap_scores_per_subject[subject] = {feature: max(0, min(1, score)) for feature, score in scores.items()}

    overall_overlap_score = max(0, min(1, overall_overlap_score))
    overall_overlap_score = (1 - overall_overlap_score) * 100

    overlap_scores_table = []
    for subject, scores in overlap_scores_per_subject.items():
        overlap_scores_table.append([subject] + [scores[feature] for feature in sorted(scores.keys())])

    overlap_feature_avg = [
    np.mean([subject_scores[feature] for subject_scores in overlap_scores_per_subject.values()])
    for feature in range(len(next(iter(overlap_scores_per_subject.values()))))
    ]

    return overlap_scores_table, overall_overlap_score, overlap_feature_avg


def process_mutual_information(eeg_labels, features=None):
    """
    Procesa la Información Mutua (MI) entre las características y las etiquetas EEG.

    Parámetros:
        eeg_labels (dict): Diccionario con las etiquetas correspondientes de los datos {sujeto: {sesión: etiquetas}}.
        features (dict): Características extraídas previamente, si ya se extrajeron.

    Retorna:
        mi_scores_table (list): Lista de listas que contiene los puntajes MI por característica y sujeto.
                                Cada lista interna tiene la forma [sujeto, score_feature_0, score_feature_1, ...].
        overall_mi_score (float): Puntaje de MI general del dataset.
    """

    # Llamar a la función que calcula MI por sujeto y obtener los puntajes
    mi_scores_per_subject, overall_mi_score = calculate_mutual_information(features, eeg_labels)

    # Limitar los scores entre 0 y 1
    for subject, score in mi_scores_per_subject.items():
        mi_scores_per_subject[subject] = max(0, min(1, score))

    overall_mi_score = max(0, min(1, overall_mi_score))
    overall_mi_score = overall_mi_score * 100

    # Crear la tabla de puntajes de MI
    mi_scores_table = []
    for subject, score in mi_scores_per_subject.items():
        mi_scores_table.append([subject, score])

    # Calcular el promedio de MI para cada característica
    mi_feature_avg = [
        np.mean([subject_scores for subject_scores in mi_scores_per_subject.values()])
    ]

    return mi_scores_table, overall_mi_score, mi_feature_avg