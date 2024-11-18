import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import welch

def extract_p300_features(eeg_data, sampling_rate=128):
    """
    Extrae características de ventanas ya generadas.

    Parámetros:
        eeg_data (dict): Diccionario con los datos EEG ya segmentados en ventanas,
                         {sujeto: {sesión: (n_canales, n_ventanas, n_muestras)}}.
        sampling_rate (int): Frecuencia de muestreo de los datos EEG.

    Retorna:
        features (dict): Diccionario con las características extraídas para cada sujeto y sesión,
                         {sujeto: {sesión: (n_canales, n_ventanas, n_features)}}.
    """
    num_features = 4  # Promedio, desviación estándar, curtosis, potencia en banda Alpha
    features = {}

    for subject, sessions in eeg_data.items():
        features[subject] = {}
        for session, data in sessions.items():
            n_canales, n_ventanas, n_muestras = data.shape
            session_features = np.zeros((n_canales, n_ventanas, num_features))

            for ch in range(n_canales):
                for win in range(n_ventanas):
                    window = data[ch, win, :]
                    avg_amplitude = np.mean(window)  # Promedio
                    std_dev = np.std(window)  # Desviación estándar
                    window_kurtosis = kurtosis(window)  # Curtosis
                    freqs, psd = welch(window, fs=sampling_rate, nperseg=min(len(window), 256))
                    alpha_band_power = np.sum(psd[(freqs >= 8) & (freqs <= 13)])  # Potencia en banda Alpha

                    session_features[ch, win, 0] = avg_amplitude
                    session_features[ch, win, 1] = std_dev
                    session_features[ch, win, 2] = window_kurtosis
                    session_features[ch, win, 3] = alpha_band_power

            features[subject][session] = session_features

    return features