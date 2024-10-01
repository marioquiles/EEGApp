import numpy as np
from scipy.signal import welch

def extract_emotional_features(eeg_data, sampling_rate=128, window_size=256, overlap=128):
    """
    Extrae características relacionadas con el estado emocional a partir de datos EEG.
    En este caso, se calculan las potencias de las bandas de frecuencia Delta, Theta, Alpha, Beta, Gamma por ventanas.

    Parámetros:
        eeg_data (dict): Diccionario con los datos EEG de dimensiones {sujeto: {sesión: (num_canales, num_muestras)}}.
        sampling_rate (int): Frecuencia de muestreo de los datos EEG.
        window_size (int): Tamaño de la ventana en muestras.
        overlap (int): Cantidad de solapamiento entre ventanas en muestras.

    Retorna:
        features (dict): Diccionario con las características extraídas para cada sujeto y sesión.
                         {sujeto: {sesión: (num_canales, num_ventanas, num_features)}}.
    """
    # Definir los límites de las bandas de frecuencia en Hz
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 45)
    }
    window_size = int(window_size)
    overlap = int(overlap)

    num_features = len(bands)
    features = {}

    # Iterar sobre cada sujeto
    for subject, sessions in eeg_data.items():
        features[subject] = {}
        
        # Iterar sobre cada sesión
        for session, data in sessions.items():
            num_channels, num_samples = data.shape
            step_size = window_size - overlap

            # Verificar que el tamaño de la ventana sea menor o igual al número de muestras
            if window_size > num_samples:
                raise ValueError(f"El tamaño de la ventana ({window_size}) es mayor que el número de muestras ({num_samples}) para la sesión {session} del sujeto {subject}.")
            
            # Calcular el número de ventanas
            num_windows = (num_samples - window_size) // step_size + 1
            if num_windows <= 0:
                raise ValueError(f"El solapamiento es demasiado grande ({overlap}) para el tamaño de la ventana ({window_size}) y el número de muestras ({num_samples}).")

            # Inicializar el array para almacenar las características de la sesión actual
            session_features = np.zeros((num_channels, num_windows, num_features))

            # Iterar sobre canales
            for ch in range(num_channels):
                # Obtener la señal del canal actual
                signal = data[ch, :]

                # Extraer características por ventana
                for i in range(num_windows):
                    start_idx = i * step_size
                    end_idx = start_idx + window_size
                    window = signal[start_idx:end_idx]

                    # Calcular la densidad espectral de potencia (PSD) usando el método de Welch
                    freqs, psd = welch(window, fs=sampling_rate, nperseg=min(window_size, len(window)))

                    # Calcular la potencia para cada banda de frecuencia
                    for j, (band, (low_freq, high_freq)) in enumerate(bands.items()):
                        # Encontrar los índices de las frecuencias dentro del rango de la banda
                        band_indices = np.logical_and(freqs >= low_freq, freqs <= high_freq)
                        # Calcular la potencia total en la banda de frecuencia
                        band_power = np.sum(psd[band_indices])
                        session_features[ch, i, j] = band_power

            features[subject][session] = session_features

    return features
