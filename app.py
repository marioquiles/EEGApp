from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, session
import os
import numpy as np
from werkzeug.utils import secure_filename
from processing.eeg_processing import process_mutual_information, process_session_metrics, process_length_metrics, process_outlier_metrics, process_labels_metrics, process_eeg_features, process_class_overlap, calculate_mutual_information, calculate_noise, process_homogeneity_and_variation
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
        return render_template('upload.html')

    if request.method == 'POST':
        compare_datasets = 'compare_datasets' in request.form
        file1 = request.files.get('file1')
        file2 = request.files.get('file2') if compare_datasets else None

        if not file1 or (compare_datasets and not file2):
            flash('Please upload the required file(s).')
            return redirect(request.url)

        # Validate files
        if not allowed_file(file1.filename) or (compare_datasets and file2 and not allowed_file(file2.filename)):
            flash('Invalid file type. Only .pkl files are allowed.')
            return redirect(request.url)

        # Save file1
        filename1 = secure_filename(file1.filename)
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        file1.save(filepath1)

        # Register file1 in file_data
        file_data[filename1] = filepath1

        if compare_datasets:
            # Save file2
            filename2 = secure_filename(file2.filename)
            filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
            file2.save(filepath2)

            # Register file2 in file_data
            file_data[filename2] = filepath2

            # Redirect to comparison parameters page
            return redirect(url_for('compare_parameters', file1=filename1, file2=filename2))

        # Redirect to parameters page for single file
        return redirect(url_for('parameters', filename=filename1))



@app.route('/parameters/<filename>', methods=['GET'])
def parameters(filename):
    return render_template('parameters.html', filename=filename)

@app.route('/get_parameters/<filename>', methods=['POST'])
def get_parameters(filename):
    # Obtener el tipo de análisis desde el formulario
    analysis_type = request.form.get('analysis_type')
    
    # Validar el tipo de análisis
    if analysis_type == 'emotions':
        return redirect(url_for('get_emotions_parameters', filename=filename))
    elif analysis_type == 'p300':
        return redirect(url_for('get_p300_parameters', filename=filename))
    else:
        return "Invalid analysis type selected.", 400

@app.route('/compare_parameters/<file1>/<file2>', methods=['GET', 'POST'])
def compare_parameters(file1, file2):
    if request.method == 'POST':
        # Recoger parámetros para el primer dataset
        params1 = {
            'sampling_frequency': float(request.form['sampling_frequency1']),
            'window_size': int(request.form['window_size1']),
            'overlap': float(request.form['overlap1']),
            'analysis_type': request.form['analysis_type1']  # 'p300' o 'emotions'
        }

        # Recoger parámetros para el segundo dataset
        params2 = {
            'sampling_frequency': float(request.form['sampling_frequency2']),
            'window_size': int(request.form['window_size2']),
            'overlap': float(request.form['overlap2']),
            'analysis_type': request.form['analysis_type2']  # 'p300' o 'emotions'
        }

        # Validar solapamiento para ambos datasets
        if params1['overlap'] >= params1['window_size'] or params2['overlap'] >= params2['window_size']:
            flash('Overlap must be less than the window size for both datasets.')
            return redirect(request.url)

        # Guardar parámetros en sesión para ser utilizados más adelante
        session['compare_params'] = {'file1': file1, 'file2': file2, 'params1': params1, 'params2': params2}

        # Redirigir al procesamiento de resultados
        return redirect(url_for('process_comparison_results', file1=file1, file2=file2))

    return render_template('compare_parameters.html', file1=file1, file2=file2)


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
        outlier_results = process_outlier_metrics(eeg_data, window_size, overlap, "emotion")
        
        # Paso 5: Calcular el ruido (SNR y eficiencia de filtrado)
        processing_status[filename] = 'Step 5: Calculating noise metrics (SNR and filtering efficiency)...'
        noise_results = calculate_noise(eeg_data, sampling_frequency)  

        # Paso 4: Calcular el desbalance de clases
        processing_status[filename] = 'Step 6: Calculating class imbalance...'
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
            feature_filepath = process_eeg_features(eeg_data, "emotions", filename, sampling_frequency, window_size, overlap)

            # Cargar las características después de extraerlas
            with open(feature_filepath, 'rb') as f:
                features = pickle.load(f)
        
        # Procesar el solapamiento de clases
        processing_status[filename] = 'Step 7: Calculating class overlap...'
        overlap_scores_table, overall_overlap_score, overlap_feature_avg = process_class_overlap(eeg_labels, "emotion", features)

        # Procesar el solapamiento de clases
        processing_status[filename] = 'Step 8: Calculating mutual information...'
        mi_scores_table, mi_overlap_score, mi_feature_avg = process_mutual_information(eeg_labels , features, "emotion")

        # Paso 10: Calcular homogeneidad y variación por sujeto
        processing_status[filename] = 'Step 10: Calculating homogeneity and subject variation...'
        homogeneity_results = process_homogeneity_and_variation(features, "emotion")

        # Combinar todos los resultados
        results = {**session_results, **length_results, **outlier_results, **imbalance_results, **noise_results, 
            'overlap_scores_table': overlap_scores_table, 'overall_overlap_score': overall_overlap_score,
            "overlap_feature_avg": overlap_feature_avg, 'mi_scores_table': mi_scores_table, 'overall_mi_score': mi_overlap_score,
            'mi_feature_avg': mi_feature_avg, **homogeneity_results}

        # Guardar los resultados en el diccionario de estado usando una nueva clave
        processing_status[f'{filename}_results'] = results
        processing_status[filename] = 'Completed'
    except Exception as e:
        processing_status[filename] = f'Error: {str(e)}'

@app.route('/get_parameters/emotions/<filename>', methods=['GET', 'POST'])
def get_emotions_parameters(filename):
    if request.method == 'POST':
        sampling_frequency = float(request.form['sampling_frequency'])
        window_size = int(request.form['window_size'])
        overlap = float(request.form['overlap'])
        session["filename"] = filename
        
        # Validar que el overlap sea menor que el tamaño de la ventana
        if overlap >= window_size:
            flash('Overlap must be less than the window size.')
            return redirect(request.url)

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

# Define the P300-specific data processing function
def process_p300_data(filepath, filename, sampling_frequency, window_size, overlap):
    try:
        # Load the EEG data
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
        
        # Extract EEG data and labels
        eeg_data = data['eeg_data']
        eeg_labels = data['eeg_labels']
        

        # Step 1: Calculate session metrics (if applicable to P300)
        processing_status[filename] = 'Step 1: Calculating session metrics...'
        session_results = process_session_metrics(eeg_data)

        # Step 2: Calculate length metrics
        processing_status[filename] = 'Step 2: Calculating length metrics...'
        length_results = process_length_metrics(eeg_data)

        processing_status[filename] = 'Step 3: Calculating window outliers...'
        outlier_results = process_outlier_metrics(eeg_data, window_size, overlap, "p300")

        processing_status[filename] = 'Step 5: Calculating noise metrics (SNR and filtering efficiency)...'
        noise_results = calculate_noise(eeg_data, sampling_frequency, "p300")  

        processing_status[filename] = 'Step 6: Calculating class imbalance...'
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
            feature_filepath = process_eeg_features(eeg_data, "p300", filename, sampling_frequency, window_size, overlap)

            # Cargar las características después de extraerlas
            with open(feature_filepath, 'rb') as f:
                features = pickle.load(f)
    

        processing_status[filename] = 'Step 7: Calculating class overlap...'
        overlap_scores_table, overall_overlap_score, overlap_feature_avg = process_class_overlap(eeg_labels, "p300", features)

        processing_status[filename] = 'Step 8: Calculating mutual information...'
        mi_scores_table, mi_overlap_score, mi_feature_avg = process_mutual_information( eeg_labels , features, "p300")
        
        processing_status[filename] = 'Step 10: Calculating homogeneity and subject variation...'
        homogeneity_results = process_homogeneity_and_variation(features, "emotion")

        # Combine all results
        results = {
            **session_results,
            **length_results,
            **outlier_results,
            **noise_results,
            **imbalance_results,
            "overall_homogeneity_score": 0,
            "overall_overlap_score": overall_overlap_score,            # Puntaje general de solapamiento de clases
            "overlap_scores_table": overlap_scores_table,            # Tabla de puntajes de solapamiento por característica y sujeto
            "overlap_feature_avg": overlap_feature_avg,             # Promedio de solapamiento por característica
            "overall_mi_score": mi_overlap_score,                 # Puntaje general de información mutua
            "mi_scores_table": mi_scores_table,                 # Tabla de puntajes de información mutua
            "mi_feature_avg": mi_feature_avg,                  # Promedio de MI por característica
            **homogeneity_results
        }

        # Save results in the processing status dictionary
        processing_status[f'{filename}_results'] = results
        processing_status[filename] = 'Completed'
    
    except Exception as e:
        processing_status[filename] = f'Error: {str(e)}'

@app.route('/get_parameters/p300/<filename>', methods=['GET', 'POST'])
def get_P300_parameters(filename):
    if request.method == 'POST':
        # Retrieve form parameters
        sampling_frequency = float(request.form['sampling_frequency'])
        window_size = int(request.form['window_size'])
        overlap = float(request.form['overlap'])
        session["filename"] = filename
        
        # Validate that overlap is less than window size
        if overlap >= window_size:
            flash('Overlap must be less than the window size.')
            return redirect(request.url)
        
        # Start processing in a new thread
        filepath = file_data.get(filename)
        if filepath:
            thread = Thread(target=process_p300_data, args=(filepath, filename, sampling_frequency, window_size, overlap))
            thread.start()
            return redirect(url_for('processing_status_view', filename=filename))
        else:
            return "File not found", 404

    return render_template('parameters.html', filename=filename)




@app.route('/results/<filename>')
def results_view(filename):
    if filename in processing_status and processing_status[filename] == 'Completed':
        # Obtener resultados del diccionario
        if ("results" not in session.keys()):
            results = processing_status.get(f'{filename}_results', {})
            print("IF")
            print(results)
            session['results'] = results

        else:        
            results = session['results'] 
            print("NO IF")
            print(results)

        rubric_score = session.get('rubric_score', 0)

        # Verificar si `results` contiene las claves necesarias
        return render_template('result.html', filename=filename, results=results)
    elif filename in processing_status and processing_status[filename].startswith('Error'):
        return f"Error processing file: {processing_status[filename]}"
    else:
        return "Processing not completed yet or unknown file", 404



@app.route('/rubrics', methods=['GET', 'POST'])
def rubrics_view():
    if request.method == 'POST':
        # Recoger los datos enviados desde el formulario, convertir a 0 si está vacío
        metadata_subject_info = float(request.form.get('metadata_subject_info', 0) or 0)
        metadata_electrode_config = float(request.form.get('metadata_electrode_config', 0) or 0)
        metadata_experimental_conditions = float(request.form.get('metadata_experimental_conditions', 0) or 0)
        metadata_recording_procedure = float(request.form.get('metadata_recording_procedure', 0) or 0)
        metadata_consistency = float(request.form.get('metadata_consistency', 0) or 0)

        format_data_format = float(request.form.get('format_data_format', 0) or 0)
        format_standardization = float(request.form.get('format_standardization', 0) or 0)
        format_conversion = float(request.form.get('format_conversion', 0) or 0)
        format_accessibility = float(request.form.get('format_accessibility', 0) or 0)

        data_compatibility = float(request.form.get('data_compatibility', 0) or 0)
        data_import_libraries = float(request.form.get('data_import_libraries', 0) or 0)
        data_loading_ease = float(request.form.get('data_loading_ease', 0) or 0)
        data_documentation = float(request.form.get('data_documentation', 0) or 0)

        relevance_objectives = float(request.form.get('relevance_objectives', 0) or 0)
        relevance_clinical_coverage = float(request.form.get('relevance_clinical_coverage', 0) or 0)
        relevance_data_specificity = float(request.form.get('relevance_data_specificity', 0) or 0)
        relevance_diversity = float(request.form.get('relevance_diversity', 0) or 0)

        # Sumar los valores de cada rúbrica
        total_metadata_score = (metadata_subject_info + metadata_electrode_config +
                                metadata_experimental_conditions + metadata_recording_procedure + metadata_consistency)

        total_format_score = (format_data_format + format_standardization +
                              format_conversion + format_accessibility)

        total_data_accessibility_score = (data_compatibility + data_import_libraries +
                                          data_loading_ease + data_documentation)

        total_relevance_score = (relevance_objectives + relevance_clinical_coverage +
                                 relevance_data_specificity + relevance_diversity)

        # Sumar todas las rúbricas
        total_score = total_metadata_score + total_format_score + total_data_accessibility_score + total_relevance_score

        # Calcular la puntuación final sobre 100 (el total máximo es 25)
        final_score = (total_score / 25) * 100

        # Guardar la puntuación en session para poder usarla en la vista de resultados
        session['rubric_score'] = final_score

        # Redirigir a la página de resultados, pasando la puntuación final
        return redirect(url_for('results_view', filename= session["filename"]))

    # Mostrar el formulario para rellenar las rúbricas
    return render_template('rubrics.html')


@app.route('/process_comparison_results/<file1>/<file2>', methods=['GET'])
def process_comparison_results(file1, file2):
    # Recuperar parámetros desde la sesión
    compare_params = session.get('compare_params', {})
    params1 = compare_params.get('params1', {})
    params2 = compare_params.get('params2', {})

    filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], file1)
    filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], file2)

    # Diccionario para almacenar resultados
    results = {}

    # Procesar Dataset 1
    if params1.get('analysis_type') == 'p300':
        process_p300_data(filepath1, file1, params1['sampling_frequency'], params1['window_size'], params1['overlap'])
    else:
        process_data(filepath1, file1, params1['sampling_frequency'], params1['window_size'], params1['overlap'])

    # Recuperar resultados del Dataset 1
    results['dataset1'] = {
        'filename': file1,
        'results': processing_status.get(f'{file1}_results', {})
    }

    # Procesar Dataset 2
    if params2.get('analysis_type') == 'p300':
        process_p300_data(filepath2, file2, params2['sampling_frequency'], params2['window_size'], params2['overlap'])
    else:
        process_data(filepath2, file2, params2['sampling_frequency'], params2['window_size'], params2['overlap'])

    # Recuperar resultados del Dataset 2
    results['dataset2'] = {
        'filename': file2,
        'results': processing_status.get(f'{file2}_results', {})
    }

    # Renderizar página de comparación
    return render_template('results_compare.html', datasets=results)


@app.route('/compare_results/<file1>/<file2>', methods=['GET'])
def compare_results_view(file1, file2):
    # Obtener resultados procesados
    results1 = processing_status.get(f'{file1}_results', {})
    results2 = processing_status.get(f'{file2}_results', {})

    return render_template('results_compare.html', file1=file1, file2=file2, results1=results1, results2=results2)

if __name__ == '__main__':
    app.run(debug=True)
