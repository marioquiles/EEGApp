import numpy as np

def calculate_homogeneity_score(eeg_features):
    """
    Calcula un puntaje de homogeneidad general basado en la desviación estándar
    de la media de las características entre todos los sujetos.
    """
    all_means = []

    for subject, sessions in eeg_features.items():
        subject_means = []
        for session_data in sessions.values():
            session_mean = np.mean(session_data)
            subject_means.append(session_mean)
        
        subject_mean = np.mean(subject_means)
        all_means.append(subject_mean)
    
    overall_std = np.std(all_means)
    overall_mean = np.mean(all_means)
    
    
    # Escalar la homogeneidad con una constante para reducir la influencia de la desviación alta
    scaling_factor = 0.1  # Ajustar según los resultados deseados
    overall_homogeneity_score = max(0, min(100, 100 * (1 - scaling_factor * (overall_std / (overall_mean + 1e-5)))))
    
    return overall_homogeneity_score

def compute_homogeneity_scores(eeg_features):
    """
    Calcula la homogeneidad y la variación por sujeto.

    Retorna:
        - overall_homogeneity_score: Puntaje único de homogeneidad para el dataset.
        - homogeneity_scores: Puntaje de homogeneidad por característica.
        - subject_variations: Variación por sujeto.
    """
    # Calcular puntaje general de homogeneidad
    overall_homogeneity_score = calculate_homogeneity_score(eeg_features)

    # Calcular homogeneidad por característica
    homogeneity_scores = {}
    for feature_idx in range(eeg_features[next(iter(eeg_features))][next(iter(eeg_features[next(iter(eeg_features))]))].shape[-1]):
        feature_values = [
            np.mean(session_data[:, :, feature_idx])
            for sessions in eeg_features.values()
            for session_data in sessions.values()
        ]
        homogeneity_scores[feature_idx] = max(0, min(100, 100 - np.std(feature_values)))

    # Calcular variaciones por sujeto
    subject_variations = {}
    all_subject_means = [
        np.mean([np.mean(session_data) for session_data in sessions.values()])
        for sessions in eeg_features.values()
    ]
    group_mean = np.mean(all_subject_means)
    group_std = np.std(all_subject_means)

    for subject, sessions in eeg_features.items():
        subject_mean = np.mean([np.mean(session_data) for session_data in sessions.values()])
        variation_score = max(0, min(100, 100 * (1 - abs(subject_mean - group_mean) / (group_std + 1e-5))))
        subject_variations[subject] = variation_score

    return {
        'overall_homogeneity_score': overall_homogeneity_score,
        'homogeneity_scores': homogeneity_scores,
        'subject_variations': subject_variations,
    }



def calculate_p300_homogeneity(eeg_features):
    """
    Calcula la homogeneidad y la variación por sujeto para datos P300.

    Retorna:
        - overall_homogeneity_score: Puntaje único de homogeneidad para el dataset.
        - homogeneity_scores: Puntaje de homogeneidad por característica.
        - subject_variations: Variación por sujeto.
    """
    # Calcular homogeneidad general
    all_subject_means = []
    for subject, sessions in eeg_features.items():
        subject_means = []
        for session_data in sessions.values():
            session_mean = np.mean(session_data)  # Media de todas las ventanas y canales
            subject_means.append(session_mean)
        
        subject_mean = np.mean(subject_means)  # Media del sujeto
        all_subject_means.append(subject_mean)

    overall_std = np.std(all_subject_means)
    overall_mean = np.mean(all_subject_means)

    # Escalar la homogeneidad con un factor para reducir impacto de alta variación
    scaling_factor = 0.1  # Ajustar si es necesario
    overall_homogeneity_score = max(0, min(100, 100 * (1 - scaling_factor * (overall_std / (overall_mean + 1e-5)))))

    # Calcular homogeneidad por característica
    homogeneity_scores = {}
    for feature_idx in range(eeg_features[next(iter(eeg_features))][next(iter(eeg_features[next(iter(eeg_features))]))].shape[-1]):
        feature_values = [
            np.mean(session_data[:, :, feature_idx])
            for sessions in eeg_features.values()
            for session_data in sessions.values()
        ]
        homogeneity_scores[feature_idx] = max(0, min(100, 100 - np.std(feature_values)))

    # Calcular variación por sujeto
    subject_variations = {}
    group_mean = np.mean(all_subject_means)
    group_std = np.std(all_subject_means)

    for subject, sessions in eeg_features.items():
        subject_mean = np.mean([np.mean(session_data) for session_data in sessions.values()])
        variation_score = max(0, min(100, 100 * (1 - abs(subject_mean - group_mean) / (group_std + 1e-5))))
        subject_variations[subject] = variation_score

    return {
        'overall_homogeneity_score': overall_homogeneity_score,
        'homogeneity_scores': homogeneity_scores,
        'subject_variations': subject_variations,
    }
