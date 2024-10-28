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
    overall_homogeneity_score = calculate_homogeneity_score(eeg_features)
    
    subject_variations = {}
    all_subject_means = [np.mean([np.mean(session_data) for session_data in sessions.values()])
                         for sessions in eeg_features.values()]
    
    group_mean = np.mean(all_subject_means)
    group_std = np.std(all_subject_means)

    for subject, sessions in eeg_features.items():
        subject_mean = np.mean([np.mean(session_data) for session_data in sessions.values()])
        variation_score = max(0, min(100, 100 * (1 - abs(subject_mean - group_mean) / (group_std + 1e-5))))
        subject_variations[subject] = variation_score

    return {
        'overall_homogeneity_score': overall_homogeneity_score,
        'subject_variations': subject_variations
    }
