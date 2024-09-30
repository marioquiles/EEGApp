from processing.calculate_sessions import calculate_sessions_expected, calculate_session_scores, calculate_final_score
from processing.calculate_length_scores import calculate_length_scores
from processing.input_validation import validate_window_size
from .calculate_outliers import calculate_outliers, calculate_outlier_score
import numpy as np

def process_eeg_data(eeg_data):
    """
    Punto de entrada para procesar los datos EEG.
    """
    # Calcular métricas relacionadas con las sesiones
    sessions_realized, sessions_expected = calculate_sessions_expected(eeg_data)
    session_scores = calculate_session_scores(sessions_realized, sessions_expected)
    final_score_sessions = calculate_final_score(session_scores)

    # Calcular métricas relacionadas con la longitud de las sesiones
    scores_per_channel, overall_score_length = calculate_length_scores(eeg_data)

    # Preparar los resultados para ser devueltos
    results = {
        'session_scores': session_scores,
        'final_score_sessions': final_score_sessions,
        'num_sujetos': len(sessions_realized),
        'sessions_expected': sessions_expected,
        'scores_per_channel': scores_per_channel,
        'overall_score_length': overall_score_length,
    }
    return results

def process_eeg_outliers(eeg_data, sampling_frequency, window_size, overlap):
    """
    Procesa los datos EEG para calcular los outliers.

    Parámetros:
        eeg_data: Array de datos EEG con dimensiones (sujetos, sesiones, canales, muestras).
        sampling_frequency: Frecuencia de muestreo del archivo.
        window_size: Tamaño de la ventana.
        overlap: Porcentaje de solapamiento entre ventanas (valor entre 0 y 1).

    Retorna:
        result: Diccionario con los resultados de outliers por sujeto.
    """
    try:
        # Calcular outliers usando la función calculate_outliers
        num_outlier_windows, total_windows = calculate_outliers(eeg_data, sampling_frequency, window_size, overlap)

        # Imprimir para depurar
        print("num_outlier_windows:", num_outlier_windows)
        print("total_windows:", total_windows)

        # Calcular el porcentaje de ventanas con outliers para cada sujeto
        percentage_outlier_windows_per_subject = []
        for subject_idx in range(len(num_outlier_windows)):
            total_outliers_for_subject = sum(num_outlier_windows[subject_idx])
            total_windows_for_subject = sum(total_windows[subject_idx])
            if total_windows_for_subject > 0:
                percentage_outliers = (total_outliers_for_subject / total_windows_for_subject) * 100
                # Calcular el puntaje de outliers usando la función ya definida
                score = calculate_outlier_score(total_outliers_for_subject, total_windows_for_subject)
            else:
                percentage_outliers = 0
                score = 0

            percentage_outlier_windows_per_subject.append({
                'subject_index': subject_idx,
                'percentage_outliers': percentage_outliers,
                'score': score
            })

        # Crear el diccionario de resultados
        result = {
            'percentage_outlier_windows': percentage_outlier_windows_per_subject
        }

        # Imprimir resultados para depuración
        print("Result:", result)

        return result

    except Exception as e:
        print(f"Error en process_eeg_outliers: {e}")
        return {'error': str(e)}
