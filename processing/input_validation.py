def validate_window_size(window_size, sampling_frequency, total_samples):
    """
    Valida el tamaño de la ventana según las restricciones establecidas.

    Parámetros:
        window_size: Tamaño de la ventana propuesto.
        sampling_frequency: Frecuencia de muestreo de los datos.
        total_samples: Longitud total de los datos (número de muestras).

    Retorna:
        bool: True si el tamaño de la ventana es válido, False en caso contrario.
    """
    min_window_size = sampling_frequency / 4
    max_window_size = total_samples

    if window_size < min_window_size:
        raise ValueError(f"Window size must be at least one fourth of the sampling frequency ({min_window_size}).")
    if window_size > max_window_size:
        raise ValueError(f"Window size must not exceed the total length of the data ({max_window_size}).")
    
    return True
