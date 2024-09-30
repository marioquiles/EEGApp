from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
import os
import numpy as np
from werkzeug.utils import secure_filename
from processing.eeg_processing import process_eeg_data, process_eeg_outliers
import time  # Para simular el tiempo de procesamiento
from threading import Thread  # Importar Thread para procesamiento en segundo plano

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['SECRET_KEY'] = 'tu_clave_secreta'

ALLOWED_EXTENSIONS = {'npz'}
processing_status = {}  # Diccionario para mantener el estado del procesamiento por archivo
file_data = {}  # Diccionario para almacenar datos de archivos subidos (frecuencia, etc.)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
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
        
        # Validar y procesar el archivo en segundo plano
        def process_data(filepath, filename, sampling_frequency, window_size, overlap):
            try:
                data = np.load(filepath)
                eeg_data = data['data']

                # Procesar otras métricas primero
                processing_status[filename] = 'Step 1: Calculating other metrics...'
                time.sleep(2)  # Simulación de procesamiento
                results = process_eeg_data(eeg_data)

                # Procesar la nueva dimensión de outliers
                processing_status[filename] = 'Step 2: Calculating window outliers...'
                time.sleep(2)  # Simulación de procesamiento
                outlier_results = process_eeg_outliers(eeg_data, sampling_frequency, window_size, overlap)

                # Añadir los resultados de outliers al diccionario de resultados
                results.update(outlier_results)

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
