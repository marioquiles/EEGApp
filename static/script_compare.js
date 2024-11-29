function toggleDetails(containerId) {
    var container = document.getElementById(containerId);
    if (container.style.display === "none" || container.style.display === "") {
        container.style.display = "block";
    } else {
        container.style.display = "none";
    }
}




// Función genérica para inicializar gráficos
function createChart(canvasId, chartType, labels, data, backgroundColor, borderColor, chartLabel) {
const canvas = document.getElementById(canvasId);
if (canvas) {
    const ctx = canvas.getContext('2d');
    console.log(`Initializing chart: ${canvasId}`);
    new Chart(ctx, {
        type: chartType,
        data: {
            labels: labels,
            datasets: [{
                label: chartLabel,
                data: data,
                backgroundColor: backgroundColor,
                borderColor: borderColor,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: true }
            },
            scales: { y: { beginAtZero: true } }
        }
    });
} else {
    console.error(`Canvas with ID '${canvasId}' not found.`);
}
}

// Inicializar gráficos después de cargar el DOM
document.addEventListener('DOMContentLoaded', function () {
    // Ejemplo: Gráfico de radar para Mutual Information Score (Dataset 1 y 2)
    createChart(
        'miRadarChart1',
        'radar',
        [
            {% for feature_index in range(1, datasets.dataset1.results.mi_scores_table[0] | length) %}
                "Feature {{ feature_index }}",
            {% endfor %}
        ],
        [
            {% for feature_avg in datasets.dataset1.results.mi_feature_avg %}
                {{ feature_avg }},
            {% endfor %}
        ],
        'rgba(75, 192, 192, 0.2)',
        'rgba(75, 192, 192, 1)',
        'Mutual Information Score'
    );

    createChart(
        'miRadarChart2',
        'radar',
        [
            {% for feature_index in range(1, datasets.dataset2.results.mi_scores_table[0] | length) %}
                "Feature {{ feature_index }}",
            {% endfor %}
        ],
        [
            {% for feature_avg in datasets.dataset2.results.mi_feature_avg %}
                {{ feature_avg }},
            {% endfor %}
        ],
        'rgba(255, 159, 64, 0.2)',
        'rgba(255, 159, 64, 1)',
        'Mutual Information Score'
    );

    // Ejemplo: Gráfico de barras para SNR (Dataset 1)
    createChart(
        'snrChart1',
        'bar',
        [
            {% for i in range(1, datasets.dataset1.results.snr_per_subject | length + 1) %}
                "Subject {{ i }}",
            {% endfor %}
        ],
        [
            {% for score in datasets.dataset1.results.snr_per_subject %}
                {{ score }},
            {% endfor %}
        ],
        'rgba(54, 162, 235, 0.6)',
        'rgba(54, 162, 235, 1)',
        'SNR (dB)'
    );

    // Ejemplo: Gráfico de barras para SNR (Dataset 2)
    createChart(
        'snrChart2',
        'bar',
        [
            {% for i in range(1, datasets.dataset2.results.snr_per_subject | length + 1) %}
                "Subject {{ i }}",
            {% endfor %}
        ],
        [
            {% for score in datasets.dataset2.results.snr_per_subject %}
                {{ score }},
            {% endfor %}
        ],
        'rgba(255, 99, 132, 0.6)',
        'rgba(255, 99, 132, 1)',
        'SNR (dB)'
    );

    createChart(
    'overlapRadarChart1', // ID del canvas
    'radar', // Tipo de gráfico
    [
        {% for feature_index in range(1, datasets.dataset1.results.overlap_scores_table[0] | length) %}
            "Feature {{ feature_index }}",
        {% endfor %}
    ], // Etiquetas
    [
        {% for feature_avg in datasets.dataset1.results.overlap_feature_avg %}
            {{ (1 - feature_avg) * 100 }},
        {% endfor %}
    ], // Datos
    'rgba(54, 162, 235, 0.2)', // Color de fondo
    'rgba(54, 162, 235, 1)', // Color del borde
    'Overlap Score' // Etiqueta del gráfico
);

createChart(
    'snrChart1', // ID del canvas
    'bar', // Tipo de gráfico
    [
        {% for i in range(1, datasets.dataset1.results.snr_per_subject | length + 1) %}
            "Subject {{ i }}",
        {% endfor %}
    ], // Etiquetas
    [
        {% for score in datasets.dataset1.results.snr_per_subject %}
            {{ score }},
        {% endfor %}
    ], // Datos
    'rgba(54, 162, 235, 0.6)', // Color de fondo
    'rgba(54, 162, 235, 1)', // Color del borde
    'SNR (dB)' // Etiqueta del gráfico
);

    createChart(
    'snrChart2', // ID del canvas
    'bar', // Tipo de gráfico
    [
        {% for i in range(1, datasets.dataset2.results.snr_per_subject | length + 1) %}
            "Subject {{ i }}",
        {% endfor %}
    ], // Etiquetas
    [
        {% for score in datasets.dataset2.results.snr_per_subject %}
            {{ score }},
        {% endfor %}
    ], // Datos
    'rgba(255, 99, 132, 0.6)', // Color de fondo
    'rgba(255, 99, 132, 1)', // Color del borde
    'SNR (dB)' // Etiqueta del gráfico
);

createChart(
    'filteringChart2', // ID del canvas
    'bar', // Tipo de gráfico
    [
        {% for i in range(1, datasets.dataset2.results.filter_efficiency_per_subject | length + 1) %}
            "Subject {{ i }}",
        {% endfor %}
    ], // Etiquetas
    [
        {% for efficiency in datasets.dataset2.results.filter_efficiency_per_subject %}
            {{ efficiency }},
        {% endfor %}
    ], // Datos
    'rgba(153, 102, 255, 0.6)', // Color de fondo
    'rgba(153, 102, 255, 1)', // Color del borde
    'Filtering Efficiency (%)' // Etiqueta del gráfico
);

createChart(
'filteringChart1', // ID del canvas
'bar', // Tipo de gráfico
[
    {% for i in range(1, datasets.dataset1.results.filter_efficiency_per_subject | length + 1) %}
        "Subject {{ i }}",
    {% endfor %}
], // Etiquetas
[
    {% for efficiency in datasets.dataset1.results.filter_efficiency_per_subject %}
        {{ efficiency }},
    {% endfor %}
], // Datos
'rgba(75, 192, 192, 0.6)', // Color de fondo
'rgba(75, 192, 192, 1)', // Color del borde
'Filtering Efficiency (%)' // Etiqueta del gráfico
);

createChart(
'imbalanceChart1', // ID del canvas
'bar', // Tipo de gráfico
[
    {% for i in range(1, datasets.dataset1.results.class_imbalance_scores | length + 1) %}
        "Subject {{ i }}",
    {% endfor %}
], // Etiquetas
[
    {% for score in datasets.dataset1.results.class_imbalance_scores %}
        {{ score }},
    {% endfor %}
], // Datos
'rgba(153, 102, 255, 0.6)', // Color de fondo
'rgba(153, 102, 255, 1)', // Color del borde
'Class Imbalance Score' // Etiqueta del gráfico
);

createChart(
'imbalanceChart2', // ID del canvas
'bar', // Tipo de gráfico
[
    {% for i in range(1, datasets.dataset2.results.class_imbalance_scores | length + 1) %}
        "Subject {{ i }}",
    {% endfor %}
], // Etiquetas
[
    {% for score in datasets.dataset2.results.class_imbalance_scores %}
        {{ score }},
    {% endfor %}
], // Datos
'rgba(255, 159, 64, 0.6)', // Color de fondo
'rgba(255, 159, 64, 1)', // Color del borde
'Class Imbalance Score' // Etiqueta del gráfico
);

createChart(
'outlierChart1', // ID del canvas
'bar', // Tipo de gráfico
[
    {% for subject in datasets.dataset1.results.outlier_percentages.keys() %}
        "Subject {{ subject }}",
    {% endfor %}
], // Etiquetas
[
    {% for percentage in datasets.dataset1.results.outlier_percentages.values() %}
        {{ percentage }},
    {% endfor %}
], // Datos
'rgba(255, 99, 132, 0.6)', // Color de fondo
'rgba(255, 99, 132, 1)', // Color del borde
'Percentage of Outlier Windows' // Etiqueta del gráfico
);

createChart(
    'lengthChart1', // ID del canvas
    'bar', // Tipo de gráfico
    [
        {% for i in range(1, datasets.dataset1.results.scores_per_channel | length + 1) %}
            "Channel {{ i }}",
        {% endfor %}
    ], // Etiquetas
    [
        {% for score in datasets.dataset1.results.scores_per_channel %}
            {{ score }},
        {% endfor %}
    ], // Datos
    'rgba(75, 192, 192, 0.6)', // Color de fondo
    'rgba(75, 192, 192, 1)', // Color del borde
    'Length Score per Channel' // Etiqueta del gráfico
);

createChart(
    'lengthChart2', // ID del canvas
    'bar', // Tipo de gráfico
    [
        {% for i in range(1, datasets.dataset2.results.scores_per_channel | length + 1) %}
            "Channel {{ i }}",
        {% endfor %}
    ], // Etiquetas
    [
        {% for score in datasets.dataset2.results.scores_per_channel %}
            {{ score }},
        {% endfor %}
    ], // Datos
    'rgba(255, 159, 64, 0.6)', // Color de fondo
    'rgba(255, 159, 64, 1)', // Color del borde
    'Length Score per Channel' // Etiqueta del gráfico
);

    createChart(
    'sessionChart1', // ID del canvas
    'bar', // Tipo de gráfico
    [
        {% for i in range(1, datasets.dataset1.results.num_sujetos + 1) %}
            "Subject {{ i }}",
        {% endfor %}
    ], // Etiquetas
    [
        {% for score in datasets.dataset1.results.session_scores %}
            {{ score }},
        {% endfor %}
    ], // Datos
    'rgba(75, 192, 192, 0.6)', // Color de fondo
    'rgba(75, 192, 192, 1)', // Color del borde
    'Session Scores' // Etiqueta del gráfico
);

createChart(
    'sessionChart2', // ID del canvas
    'bar', // Tipo de gráfico
    [
        {% for i in range(1, datasets.dataset2.results.num_sujetos + 1) %}
            "Subject {{ i }}",
        {% endfor %}
    ], // Etiquetas
    [
        {% for score in datasets.dataset2.results.session_scores %}
            {{ score }},
        {% endfor %}
    ], // Datos
    'rgba(255, 159, 64, 0.6)', // Color de fondo
    'rgba(255, 159, 64, 1)', // Color del borde
    'Session Scores' // Etiqueta del gráfico
);

});