from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, session
import os
import numpy as np
from werkzeug.utils import secure_filename
from processing.eeg_processing import process_mutual_information, process_session_metrics, process_length_metrics, process_outlier_metrics, process_labels_metrics, process_eeg_features, process_class_overlap, calculate_mutual_information, calculate_noise, process_homogeneity_and_variation
from threading import Thread  # Importar Thread para procesamiento en segundo plano
import pickle
from pathlib import Path


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['SECRET_KEY'] = 'tu_clave_secreta'

ALLOWED_EXTENSIONS = {'pkl'}
processing_status = {}  # Diccionario para mantener el estado del procesamiento por archivo
file_data = {}  # Diccionario para almacenar datos de archivos subidos (frecuencia, etc.)

PRECOMPUTED_RESULTS_DIR = "precomputed_results"
os.makedirs(PRECOMPUTED_RESULTS_DIR, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    session.pop('results', None)
    return render_template('index.html')


@app.route('/datasets', methods=['GET'])
def list_datasets():
    try:
        preloaded_datasets = [
            f for f in os.listdir(PRECOMPUTED_RESULTS_DIR) if f.endswith('.pkl')
        ]
        return jsonify({'preloaded': preloaded_datasets})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        global processing_status
        processing_status = {}  # Reinicia el diccionario
        return render_template('upload.html')

    results = {}

    if request.method == 'POST':
        compare_datasets = 'compare_datasets' in request.form
        preloaded1 = request.form.get('preloaded1')
        preloaded2 = request.form.get('preloaded2') if compare_datasets else None
        file1 = request.files.get('file1')
        file2 = request.files.get('file2') if compare_datasets else None

        # Caso 1: Todos los datasets son precargados
        if preloaded1 and (not compare_datasets or (compare_datasets and preloaded2)):
            # Cargar resultados precargados
            if preloaded1:
                filepath1 = os.path.join(PRECOMPUTED_RESULTS_DIR, preloaded1)
                if os.path.exists(filepath1):
                    with open(filepath1, 'rb') as f:
                        results['dataset1'] = {'filename': preloaded1, 'results': pickle.load(f)}
                else:
                    flash(f"Preloaded dataset {preloaded1} not found.")
                    return redirect(request.url)

            if preloaded2:
                filepath2 = os.path.join(PRECOMPUTED_RESULTS_DIR, preloaded2)
                if os.path.exists(filepath2):
                    with open(filepath2, 'rb') as f:
                        results['dataset2'] = {'filename': preloaded2, 'results': pickle.load(f)}
                else:
                    flash(f"Preloaded dataset {preloaded2} not found.")
                    return redirect(request.url)

            # Redirigir a la página de resultados precargados
            if compare_datasets:
                return render_template('results_compare.html', datasets=results)
            else:
                return render_template('result.html', filename=preloaded1, results=results['dataset1']['results'])


        # Caso 2: Un dataset subido por el usuario
        if file1 and not compare_datasets:
            # Validar archivo subido
            if not allowed_file(file1.filename):
                flash('Invalid file type. Only .pkl files are allowed.')
                return redirect(request.url)

            # Guardar el archivo subido
            filename1 = secure_filename(file1.filename)
            filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
            file1.save(filepath1)
            file_data[filename1] = filepath1

            # Redirigir a `parameters.html` para leer parámetros del dataset subido
            return redirect(url_for('parameters', filename=filename1))
        
       # Caso 3: Comparación de un dataset subido y uno precargado
        if (file1 and preloaded2) or (file2 and preloaded1):
            if file1 and preloaded2:
                user_file = file1
                preloaded_file = preloaded2
            elif file2 and preloaded1:
                user_file = file2
                preloaded_file = preloaded1

            # Validar el archivo subido
            if not allowed_file(user_file.filename):
                flash('Invalid file type. Only .pkl files are allowed.')
                return redirect(request.url)

            # Guardar el archivo subido
            user_filename = secure_filename(user_file.filename)
            user_filepath = os.path.join(app.config['UPLOAD_FOLDER'], user_filename)
            user_file.save(user_filepath)

            # Registrar en file_data
            file_data[user_filename] = user_filepath

            # Redirigir a `get_parameters_for_comparison`
            return redirect(url_for('get_parameters_for_comparison', filename=user_filename, preloaded_file=preloaded_file))



        # Caso 4: Dos datasets subidos por el usuario
        if file1 and file2:
            # Validar archivos subidos
            if not allowed_file(file1.filename) or not allowed_file(file2.filename):
                flash('Invalid file type. Only .pkl files are allowed.')
                return redirect(request.url)

            # Guardar los archivos subidos
            filename1 = secure_filename(file1.filename)
            filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
            file1.save(filepath1)
            file_data[filename1] = filepath1

            filename2 = secure_filename(file2.filename)
            filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
            file2.save(filepath2)
            file_data[filename2] = filepath2

            # Redirigir a la página de parámetros para comparación
            return redirect(url_for('compare_parameters', file1=filename1, file2=filename2))

        # Si no se cumple ninguna de las condiciones, devolver un mensaje de error
        flash('Please select or upload the required datasets.')
        return redirect(request.url)


@app.route('/process_comparison_with_preloaded/<file1>/<preloaded_file>', methods=['GET'])
def process_comparison_with_preloaded(file1, preloaded_file):
    # Recuperar parámetros del dataset subido desde la sesión
    compare_params = session.get('compare_params', {})
    params1 = compare_params.get('params1', {})

    # Construir las rutas de los archivos
    filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], file1)
    preloaded_filepath = os.path.join(PRECOMPUTED_RESULTS_DIR, preloaded_file)

    # Diccionario para almacenar resultados
    results = {}

    # Procesar el dataset subido
    if params1.get('analysis_type') == 'p300':
        process_p300_data(filepath1, file1, params1['sampling_frequency'], params1['window_size'], params1['overlap'])
    else:
        process_data(filepath1, file1, params1['sampling_frequency'], params1['window_size'], params1['overlap'])

    # Recuperar resultados del dataset subido
    results['dataset1'] = {
        'filename': file1,
        'results': processing_status.get(f'{file1}_results', {})
    }

    # Recuperar resultados del dataset precargado
    if os.path.exists(preloaded_filepath):
        with open(preloaded_filepath, 'rb') as f:
            preloaded_results = pickle.load(f)
        results['dataset2'] = {
            'filename': preloaded_file,
            'results': preloaded_results
        }
    else:
        return f"Preloaded dataset {preloaded_file} not found.", 404

    # Renderizar la página de comparación
    return render_template('results_compare.html', datasets=results)


@app.route('/get_parameters_for_comparison/<filename>/<preloaded_file>', methods=['GET', 'POST'])
def get_parameters_for_comparison(filename, preloaded_file):
    if request.method == 'GET':
        # Mostrar formulario de parámetros para el dataset subido
        return render_template('parameters_preloaded.html', filename=filename, preloaded_file=preloaded_file)

    if request.method == 'POST':
        # Obtener los parámetros del formulario
        analysis_type = request.form.get('analysis_type')
        sampling_frequency = float(request.form.get('sampling_frequency', 0))
        window_size = int(request.form.get('window_size', 0))
        overlap = float(request.form.get('overlap', 0))

        # Validar solapamiento
        if overlap >= window_size:
            flash('Overlap must be less than the window size.')
            return redirect(url_for('get_parameters_for_comparison', filename=filename, preloaded_file=preloaded_file))

        # Guardar los parámetros en la sesión
        session['compare_params'] = {
            'params1': {
                'analysis_type': analysis_type,
                'sampling_frequency': sampling_frequency,
                'window_size': window_size,
                'overlap': overlap
            }
        }

        # Redirigir a `process_comparison_with_preloaded`
        return redirect(url_for('process_comparison_with_preloaded', file1=filename, preloaded_file=preloaded_file))




@app.route('/get_parameters/<filename>', methods=['GET'])
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

        eeg_data = None
        eeg_labels = None

        for key in data.keys():
            if "label" in key.lower():  # Buscar claves que contengan 'label'
                eeg_labels = data[key]
            elif "data" in key.lower():  # Buscar claves que contengan 'data'
                eeg_data = data[key]

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
        processing_status[filename] = 'Step 4: Calculating noise metrics (SNR and filtering efficiency)...'
        noise_results = calculate_noise(eeg_data, sampling_frequency)  

        # Paso 4: Calcular el desbalance de clases
        processing_status[filename] = 'Step 5: Calculating class imbalance...'
        imbalance_results = process_labels_metrics(eeg_labels)

        base_filename = Path(filename).stem
        feature_filepath = Path("features") / f"{base_filename}_ws{window_size}_ol{int(overlap)}_features.pkl"

        # Verificar si ya existen las características extraídas
        if os.path.exists(feature_filepath):
            print(f"Características encontradas en {feature_filepath}. Cargando...")
            with open(feature_filepath, 'rb') as f:
                features = pickle.load(f)
        else:
            # Extraer y guardar las características
            processing_status[filename] = 'Step 6: Extracting features'
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
            'mi_feature_avg': mi_feature_avg, **homogeneity_results, 'rubric_score': 0}

        with open("Dreamer.pkl", "wb") as f:
            pickle.dump(results, f)
            
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
        eeg_data = None
        eeg_labels = None

        for key in data.keys():
            if "label" in key.lower():  # Buscar claves que contengan 'label'
                eeg_labels = data[key]
            elif "data" in key.lower():  # Buscar claves que contengan 'data'
                eeg_data = data[key]
        

        # Step 1: Calculate session metrics (if applicable to P300)
        processing_status[filename] = 'Step 1: Calculating session metrics...'
        session_results = process_session_metrics(eeg_data)

        # Step 2: Calculate length metrics
        processing_status[filename] = 'Step 2: Calculating length metrics...'
        length_results = process_length_metrics(eeg_data)

        processing_status[filename] = 'Step 3: Calculating window outliers...'
        outlier_results = process_outlier_metrics(eeg_data, window_size, overlap, "p300")

        processing_status[filename] = 'Step 4: Calculating noise metrics (SNR and filtering efficiency)...'
        noise_results = calculate_noise(eeg_data, sampling_frequency, "p300")  

        processing_status[filename] = 'Step 5: Calculating class imbalance...'
        imbalance_results = process_labels_metrics(eeg_labels)

        base_filename = Path(filename).stem
        feature_filepath = Path("features") / f"{base_filename}_features.pkl"
        print(feature_filepath)
        # Verificar si ya existen las características extraídas
        if os.path.exists(feature_filepath):
            print(f"Características encontradas en {feature_filepath}. Cargando...")
            with open(feature_filepath, 'rb') as f:
                features = pickle.load(f)
        else:
            # Extraer y guardar las características
            processing_status[filename] = 'Step 6: Extracting features'
            feature_filepath = process_eeg_features(eeg_data, "p300", filename, sampling_frequency, window_size, overlap)

            # Cargar las características después de extraerlas
            with open(feature_filepath, 'rb') as f:
                features = pickle.load(f)
    

        processing_status[filename] = 'Step 7: Calculating class overlap...'
        overlap_scores_table, overall_overlap_score, overlap_feature_avg = process_class_overlap(eeg_labels, "p300", features)

        processing_status[filename] = 'Step 8: Calculating mutual information...'
        mi_scores_table, mi_overlap_score, mi_feature_avg = process_mutual_information( eeg_labels , features, "p300")
        
        processing_status[filename] = 'Step 9: Calculating homogeneity and subject variation...'
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
            **homogeneity_results,
            'rubric_score': 0
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
            session['results'] = results
            if('rubric_score' in session):
                rubric_score = session.get('rubric_score', 0)
                results["rubric_score"] = rubric_score

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
