from scipy.signal import butter, filtfilt
import numpy as np

def bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=128, order=5):
    """
    Aplica un filtro pasa-bandas a los datos EEG.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=1)
    
    
    return filtered_data

def calculate_snr(eeg_data, fs=128, default_snr=-10):
    """
    Calcula el SNR para los datos de EEG.
    eeg_data: datos EEG en formato (n_canales, n_muestras).
    fs: frecuencia de muestreo.
    default_snr: valor predeterminado de SNR para casos en los que no sea posible calcularlo.
    """
    try:        
        # Filtrar la señal deseada
        filtered_signal = bandpass_filter(eeg_data, fs=fs)
        
        # Calcular la potencia de la señal en el rango deseado (0.5 - 50 Hz)
        signal_power = np.mean(np.square(filtered_signal), axis=1)
        
        # Calcular la potencia total (incluyendo ruido)
        total_power = np.mean(np.square(eeg_data), axis=1)
        
        # Calcular la potencia del ruido como la diferencia entre la potencia total y la potencia de la señal
        noise_power = total_power - signal_power
        
        # Corregir potencia del ruido negativa o cercana a cero
        noise_power = np.where(noise_power <= 0, np.nan, noise_power)
        
        # Evitar división por cero o valores NaN en el SNR
        with np.errstate(divide='ignore', invalid='ignore'):
            snr_per_channel = 10 * np.log10(np.where(noise_power > 0, signal_power / noise_power, np.nan))
        
        # Reemplazar cualquier NaN resultante con un valor por defecto
        snr_per_channel = np.where(np.isnan(snr_per_channel), default_snr, snr_per_channel)
        
        # Devolver el promedio de SNR por canal, manejando valores negativos
        snr_avg = np.nanmean(snr_per_channel)
        if snr_avg < 0:  # Si el promedio es negativo, devolver un valor alto por defecto
            snr_avg = default_snr
        
        return snr_avg

    except Exception as e:
        print(f"Error calculando SNR: {e}")
        return default_snr
    

def calculate_filtering_efficiency(eeg_data, fs=128):
    """
    Calcula la eficiencia del filtrado.
    eeg_data: datos EEG en formato (n_canales, n_muestras).
    fs: frecuencia de muestreo.
    """
    # Potencia de ruido antes del filtrado
    total_power_before = np.mean(np.square(eeg_data), axis=1)
    
    # Filtrar la señal
    filtered_signal = bandpass_filter(eeg_data, fs=fs)
    
    # Potencia de ruido después del filtrado
    signal_power_after = np.mean(np.square(filtered_signal), axis=1)
    
    # Potencia de ruido antes y después del filtrado
    noise_power_before = total_power_before - signal_power_after
    
    filtering_efficiency = (1 - (noise_power_before / total_power_before)) * 100
    
    return np.mean(filtering_efficiency)  # Devolvemos la eficiencia promedio
