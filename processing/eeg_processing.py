from processing.calculate_sessions import calculate_sessions_expected, calculate_session_scores, calculate_final_score
from processing.input_validation import validate_window_size
from processing.calculate_length_scores import calculate_length_scores
from processing.calculate_labels import calculate_class_imbalance_score
from .calculate_outliers import calculate_outliers, calculate_P300outliers
from processing.extract_emotional_features import extract_emotional_features
from processing.calculate_overlap import calculate_class_overlap, calculate_p300_class_overlap
from processing.mutual_information import calculate_mutual_information, calculate_p300_mutual_information
from processing.calculate_noise import calculate_snr, calculate_filtering_efficiency
from processing.calculate_variability_subjects import compute_homogeneity_scores, calculate_p300_homogeneity
from processing.extract_p300_features import extract_p300_features
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

def process_outlier_metrics(eeg_data, window_size, overlap, processingType):
        try:
            if (processingType == "emotion"):
                outlier_percentages = calculate_outliers(eeg_data, window_size, overlap)
            else:
                outlier_percentages = calculate_P300outliers(eeg_data)

            # Calcular la puntuación final para los outliers
            mean_outlier_percentage = np.mean(list(outlier_percentages.values()))
            final_outlier_score = max(0, 100 - mean_outlier_percentage)
            final_outlier_score = round(final_outlier_score, 2)

            # Preparar el resultado final
            result = {
                'outlier_percentages': outlier_percentages,
                'final_outlier_score': final_outlier_score
            }

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
def process_eeg_features(eeg_data, processingType, filename, sampling_rate=128, window_size=256, overlap=128, output_folder="features/"):
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
    if(processingType == "emotions"):
        features = extract_emotional_features(eeg_data, sampling_rate, window_size, overlap)
    else:
        features = extract_p300_features(eeg_data, sampling_rate)

    # Crear la carpeta de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Guardar las características extraídas en un archivo
    feature_filepath = os.path.join(output_folder, filename + "_features")
    with open(feature_filepath, 'wb') as f:
        pickle.dump(features, f)

    print(f"Características extraídas y guardadas en {feature_filepath}")
    return feature_filepath

def process_class_overlap(eeg_labels, analysis_type, features=None):
    """
    Procesa el solapamiento entre clases usando las características extraídas.
    
    Parámetros:
        eeg_labels (dict): Diccionario con las etiquetas correspondientes de los datos {sujeto: {sesión: etiquetas}}.
        analysis_type (str): Tipo de análisis, puede ser "emotion" o "p300".
        features (dict): Características extraídas previamente, si ya se extrajeron.

    Retorna:
        overlap_scores_table (list): Lista de listas que contiene los puntajes de solapamiento por característica y sujeto.
                                     Cada lista interna tiene la forma [sujeto, score_feature_0, score_feature_1, ...].
        overall_overlap_score (float): Puntaje de solapamiento general del dataset.
        overlap_feature_avg (list): Lista con los promedios de solapamiento por característica.
    """
    if analysis_type == "emotion":
        overlap_scores_per_subject, overall_overlap_score = calculate_class_overlap(features, eeg_labels)
    elif analysis_type == "p300":
        overlap_scores_per_subject, overall_overlap_score = calculate_p300_class_overlap(features, eeg_labels)
    else:
        raise ValueError(f"Analysis type '{analysis_type}' is not supported.")

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


def process_mutual_information(eeg_labels, features=None, analysis_type="emotion"):
  
    if analysis_type == "p300":
        # Usar la función específica para P300
        mi_scores_table, overall_mi_score, mi_feature_avg = calculate_p300_mutual_information(features, eeg_labels)
    else:
        # Usar la función general
        mi_scores_table, overall_mi_score, mi_feature_avg = calculate_mutual_information(features, eeg_labels)

    # Limitar los scores entre 0 y 1
    for row in mi_scores_table:
        for i in range(1, len(row)):  # Limitar cada puntaje de característica
            row[i] = max(0, min(1, row[i]))

    overall_mi_score = max(0, min(1, overall_mi_score))
    overall_mi_score = overall_mi_score * 100  # Escalar al rango [0, 100]

    # Truncar los promedios de MI a dos decimales
    mi_feature_avg = [round(avg, 2) for avg in mi_feature_avg]

    return mi_scores_table, overall_mi_score, mi_feature_avg


def calculate_noise(all_eeg_data, fs, dataType = "emotion"):

    noise_results = {
        'snr_per_subject': [],
        'filter_efficiency_per_subject': [],
        'overall_snr': 0,
        'overall_filtering_efficiency': 0
    }

    total_snr = 0
    total_filtering_efficiency = 0
    num_sessions = 0

    # Iterar sobre todos los sujetos
    for subject_id, sessions in all_eeg_data.items():
        subject_snr = []
        subject_filter_efficiency = []

        # Iterar sobre todas las sesiones de cada sujeto
        for session_id, eeg_data in sessions.items():
            # Calcular el SNR

            if (dataType == "p300"):
                eeg_data = eeg_data.reshape(eeg_data.shape[0], -1)

            snr_value = calculate_snr(eeg_data, fs)
            subject_snr.append(snr_value)

            # Calcular la eficiencia de filtrado
            filter_efficiency_value = calculate_filtering_efficiency(eeg_data, fs)
            subject_filter_efficiency.append(filter_efficiency_value)

            # Sumar al total para calcular los promedios generales
            total_snr += snr_value
            total_filtering_efficiency += filter_efficiency_value
            num_sessions += 1
        
        # Guardar los resultados por sujeto
        noise_results['snr_per_subject'].append(sum(subject_snr) / len(subject_snr))
        noise_results['filter_efficiency_per_subject'].append(sum(subject_filter_efficiency) / len(subject_filter_efficiency))
    
    # Calcular los promedios generales
    if num_sessions > 0:
        noise_results['overall_snr'] = total_snr / num_sessions
        noise_results['overall_filtering_efficiency'] = total_filtering_efficiency / num_sessions
    
    return noise_results





def process_homogeneity_and_variation(eeg_features, analysis_type="emotion"):
    """
    Calcula puntajes de homogeneidad y variación para diferentes tipos de datos EEG.

    Parámetros:
        eeg_features (dict): Características EEG extraídas por sujeto y sesión.
        analysis_type (str): Tipo de análisis ("general" o "p300").

    Retorna:
        results (dict): Diccionario que contiene:
                        - overall_homogeneity_score: Puntaje único de homogeneidad para el dataset.
                        - homogeneity_scores: Puntajes de homogeneidad por característica.
                        - subject_variations: Variaciones por sujeto.
    """
    if analysis_type == "p300":
        return calculate_p300_homogeneity(eeg_features)
    else:
        return compute_homogeneity_scores(eeg_features)
