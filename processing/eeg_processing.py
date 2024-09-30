from processing.calculate_sessions import calculate_sessions_expected, calculate_session_scores, calculate_final_score
from processing.input_validation import validate_window_size
from processing.calculate_length_scores import calculate_length_scores
from processing.calculate_labels import calculate_class_imbalance_score
from .calculate_outliers import calculate_outliers, calculate_outlier_score
import numpy as np

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
