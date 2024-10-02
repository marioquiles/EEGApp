from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
import os
import numpy as np
from werkzeug.utils import secure_filename
from processing.eeg_processing import process_session_metrics, process_length_metrics, process_outlier_metrics, process_labels_metrics, process_eeg_features, process_class_overlap
from threading import Thread  # Importar Thread para procesamiento en segundo plano
import pickle

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['SECRET_KEY'] = 'tu_clave_secreta'

ALLOWED_EXTENSIONS = {'pkl'}
processing_status = {}  # Diccionario para mantener el estado del procesamiento por archivo
file_data = {}  # Diccionario para almacenar datos de archivos subidos (frecuencia, etc.)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        # Mostrar el formulario de subida de archivos
        return render_template('upload.html')

    # Procesar la solicitud POST cuando se sube el archivo
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Inicializar el estado del procesamiento
            processing_status[filename] = 'Uploading complete. Please provide additional parameters.'

            # Almacenar el archivo subido para su posterior procesamiento
            file_data[filename] = filepath

            # Redirigir al formulario para obtener la frecuencia de muestreo y tamaño de ventana
            return redirect(url_for('get_parameters', filename=filename))
        else:
            flash('File not allowed')
            return redirect(request.url)


@app.route('/get_parameters/<filename>', methods=['GET', 'POST'])
def get_parameters(filename):
    if request.method == 'POST':
        sampling_frequency = float(request.form['sampling_frequency'])
        window_size = int(request.form['window_size'])
        overlap = float(request.form['overlap'])
        
                # Validar que el overlap sea menor que el tamaño de la ventana
        if overlap >= window_size:
            flash('Overlap must be less than the window size.')
            return redirect(request.url)
        
        # Validar y procesar el archivo en segundo plano
        def process_data(filepath, filename, sampling_frequency, window_size, overlap):
            try:
                # Abrir el archivo con pickle
                with open(filepath, 'rb') as file:
                    data = pickle.load(file)

                # Extraer los datos del diccionario
                eeg_data = data['eeg_data']
                eeg_labels = data['eeg_labels']

                # Procesar métricas relacionadas con las sesiones
                processing_status[filename] = 'Step 1: Calculating session metrics...'
                session_results = process_session_metrics(eeg_data)

                # Procesar métricas relacionadas con la longitud de las sesiones
                processing_status[filename] = 'Step 2: Calculating length metrics...'
                length_results = process_length_metrics(eeg_data)

                # Procesar la nueva dimensión de outliers
                processing_status[filename] = 'Step 3: Calculating window outliers...'
                outlier_results = process_outlier_metrics(eeg_data, window_size, overlap)

                # Paso 4: Calcular el desbalance de clases
                processing_status[filename] = 'Step 4: Calculating class imbalance...'
                imbalance_results = process_labels_metrics(eeg_labels)

                base_filename, _ = os.path.splitext(filename)
                feature_filepath = f"features/{base_filename}_features.pkl"

                # Verificar si ya existen las características extraídas
                if os.path.exists(feature_filepath):
                    print(f"Características encontradas en {feature_filepath}. Cargando...")
                    with open(feature_filepath, 'rb') as f:
                        features = pickle.load(f)
                else:
                    # Extraer y guardar las características
                    processing_status[filename] = 'Step 5: Extracting features'
                    feature_filepath = process_eeg_features(eeg_data, filename, sampling_frequency, window_size, overlap)

                    # Cargar las características después de extraerlas
                    with open(feature_filepath, 'rb') as f:
                        features = pickle.load(f)
                
                # Procesar el solapamiento de clases
                processing_status[filename] = 'Step 6: Calculating class overlap...'
                overlap_scores_table, overall_overlap_score, overlap_feature_avg = process_class_overlap(eeg_labels, features)


                # Combinar todos los resultados
                results = {**session_results, **length_results, **outlier_results, **imbalance_results, 'overlap_scores_table': overlap_scores_table,
                    'overall_overlap_score': overall_overlap_score, "overlap_feature_avg": overlap_feature_avg}

                # Guardar los resultados en el diccionario de estado usando una nueva clave
                processing_status[f'{filename}_results'] = results
                processing_status[filename] = 'Completed'
            except Exception as e:
                processing_status[filename] = f'Error: {str(e)}'

        # Ejecutar el procesamiento en un hilo separado
        filepath = file_data.get(filename)
        if filepath:
            thread = Thread(target=process_data, args=(filepath, filename, sampling_frequency, window_size, overlap))
            thread.start()
            return redirect(url_for('processing_status_view', filename=filename))
        else:
            return "File not found", 404

    return render_template('parameters.html', filename=filename)

@app.route('/process_status/<filename>')
def processing_status_view(filename):
    return render_template('loading.html', filename=filename)

@app.route('/status/<filename>', methods=['GET'])
def get_status(filename):
    status = processing_status.get(filename, 'Unknown file')
    return jsonify({'status': status})

@app.route('/results/<filename>')
def results_view(filename):
    if filename in processing_status and processing_status[filename] == 'Completed':
        # Obtener resultados del diccionario
        results = processing_status.get(f'{filename}_results', {})

        # Verificar si `results` contiene las claves necesarias
        return render_template('result.html', filename=filename, results=results)
    elif filename in processing_status and processing_status[filename].startswith('Error'):
        return f"Error processing file: {processing_status[filename]}"
    else:
        return "Processing not completed yet or unknown file", 404

if __name__ == '__main__':
    app.run(debug=True)
