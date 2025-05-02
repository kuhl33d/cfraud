document.addEventListener('DOMContentLoaded', function() {
    // Initialize charts
    fetchDataAndRenderCharts();
    
    // Set up event listeners for detection buttons
    document.querySelectorAll('.detect-btn').forEach(button => {
        button.addEventListener('click', function() {
            const rowIndex = this.getAttribute('data-index');
            detectAnomaly(rowIndex);
        });
    });
    
    // Set up event listener for custom analysis
    document.getElementById('analyzeBtn').addEventListener('click', function() {
        const rowIndex = document.getElementById('rowIndex').value;
        detectAnomaly(rowIndex);
    });
});

function fetchDataAndRenderCharts() {
    fetch('/data')
        .then(response => response.json())
        .then(data => {
            renderAmountChart(data.fraud_by_amount);
            renderCorrelationChart(data.correlations);
        })
        .catch(error => console.error('Error fetching data:', error));
}

function renderAmountChart(data) {
    const ctx = document.getElementById('amountChart').getContext('2d');
    
    const chartData = {
        labels: ['Mean', 'Median', 'Max'],
        datasets: [
            {
                label: 'Normal Transactions',
                backgroundColor: 'rgba(40, 167, 69, 0.7)',
                data: [data[0].mean, data[0].median, data[0].max]
            },
            {
                label: 'Fraudulent Transactions',
                backgroundColor: 'rgba(220, 53, 69, 0.7)',
                data: [data[1].mean, data[1].median, data[1].max]
            }
        ]
    };
    
    new Chart(ctx, {
        type: 'bar',
        data: chartData,
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Amount ($)'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Transaction Amount Statistics by Class'
                }
            }
        }
    });
}

function renderCorrelationChart(correlations) {
    const ctx = document.getElementById('correlationChart').getContext('2d');
    
    // Get top 10 correlated features (by absolute value)
    const sortedFeatures = Object.entries(correlations)
        .filter(([key]) => key !== 'Class') // Exclude Class itself
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .slice(0, 10);
    
    const labels = sortedFeatures.map(item => item[0]);
    const values = sortedFeatures.map(item => item[1]);
    const backgroundColors = values.map(value => 
        value >= 0 ? 'rgba(40, 167, 69, 0.7)' : 'rgba(220, 53, 69, 0.7)'
    );
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Correlation with Fraud',
                data: values,
                backgroundColor: backgroundColors
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Correlation Coefficient'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Top 10 Features Correlated with Fraud'
                }
            }
        }
    });
}

function detectAnomaly(rowIndex) {
    fetch('/detect', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ row_index: rowIndex })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showError(data.error);
            return;
        }
        displayResults(data);
    })
    .catch(error => {
        console.error('Error:', error);
        showError('An error occurred while processing your request.');
    });
}

function displayResults(data) {
    const resultsDiv = document.getElementById('detection-results');
    const isPredictionFraud = data.prediction.includes('Fraud');
    const isActualFraud = data.actual_class.includes('Fraud');
    
    // Create result HTML
    let resultHTML = `
        <div class="result-card ${isPredictionFraud ? 'fraud' : 'normal'}">
            <h4>Analysis Results</h4>
            <div class="prediction-badge ${isPredictionFraud ? 'fraud' : 'normal'}">
                ${data.prediction}
            </div>
            <p>Actual class: <strong>${data.actual_class}</strong></p>
            <hr>
            <h5>Transaction Details:</h5>
            <div class="row">
                <div class="col-md-6">
                    <p>Time: <span class="feature-value">${data.row_data.Time}</span></p>
                    <p>Amount: <span class="feature-value">$${data.row_data.Amount}</span></p>
                </div>
                <div class="col-md-6">
                    <p>V1: <span class="feature-value">${data.row_data.V1.toFixed(4)}</span></p>
                    <p>V2: <span class="feature-value">${data.row_data.V2.toFixed(4)}</span></p>
                </div>
            </div>
            <p class="mt-3">${isPredictionFraud === isActualFraud ? 
                '<span class="text-success">✓ Model prediction matches actual class</span>' : 
                '<span class="text-danger">✗ Model prediction differs from actual class</span>'}</p>
        </div>
    `;
    
    resultsDiv.innerHTML = resultHTML;
}

function showError(message) {
    const resultsDiv = document.getElementById('detection-results');
    resultsDiv.innerHTML = `
        <div class="alert alert-danger">
            <strong>Error:</strong> ${message}
        </div>
    `;
}