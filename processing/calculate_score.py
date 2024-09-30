def calculate_outlier_score(num_outlier_windows, total_windows):
    """
    Calcula el score basado en la proporción de ventanas normales y con outliers.

    Parámetros:
        num_outlier_windows: Número de ventanas con valores atípicos
        total_windows: Número total de ventanas

    Retorna:
        score: Puntaje en un rango de [0-100]
    """
    proportion_normals = (total_windows - num_outlier_windows) / total_windows
    proportion_outliers = num_outlier_windows / total_windows
    score = (proportion_normals - proportion_outliers) * 100

    return score
