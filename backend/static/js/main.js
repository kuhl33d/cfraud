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

    // Initialize pagination
    initPagination();
    
    // Initialize file upload functionality
    initFileUpload();
});

// File upload variables
let currentDatasetId = null;

function initFileUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadPrompt = document.getElementById('uploadPrompt');
    const uploadProgress = document.getElementById('uploadProgress');
    const progressBar = document.getElementById('progressBar');
    const uploadStatus = document.getElementById('uploadStatus');
    const uploadError = document.getElementById('uploadError');
    const uploadSuccess = document.getElementById('uploadSuccess');
    const analyzeDatasetBtn = document.getElementById('analyzeDatasetBtn');
    const clearDatasetBtn = document.getElementById('clearDatasetBtn');
    
    // Click on upload area to trigger file input
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });
    
    // Handle file selection
    fileInput.addEventListener('change', function() {
        if (fileInput.files.length > 0) {
            uploadFile(fileInput.files[0]);
        }
    });
    
    // Handle drag and drop
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', function() {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        if (e.dataTransfer.files.length > 0) {
            uploadFile(e.dataTransfer.files[0]);
        }
    });
    
    // Handle analyze dataset button
    analyzeDatasetBtn.addEventListener('click', function() {
        if (currentDatasetId) {
            analyzeDataset(currentDatasetId);
        }
    });
    
    // Handle clear dataset button
    clearDatasetBtn.addEventListener('click', function() {
        if (currentDatasetId) {
            clearDataset(currentDatasetId);
        }
    });
    
    function uploadFile(file) {
        // Check if file is CSV
        if (!file.name.endsWith('.csv')) {
            showUploadError('Only CSV files are allowed');
            return;
        }
        
        // Check file size (max 50MB)
        if (file.size > 50 * 1024 * 1024) {
            showUploadError('File size exceeds the 50MB limit');
            return;
        }
        
        // Show upload progress
        uploadPrompt.classList.add('d-none');
        uploadProgress.classList.remove('d-none');
        uploadError.classList.add('d-none');
        uploadSuccess.classList.add('d-none');
        
        // Create FormData
        const formData = new FormData();
        formData.append('file', file);
        
        // Create XMLHttpRequest for progress tracking
        const xhr = new XMLHttpRequest();
        
        // Track upload progress
        xhr.upload.addEventListener('progress', function(e) {
            if (e.lengthComputable) {
                const percentComplete = Math.round((e.loaded / e.total) * 100);
                progressBar.style.width = percentComplete + '%';
                progressBar.textContent = percentComplete + '%';
                uploadStatus.textContent = `Uploading: ${formatFileSize(e.loaded)} of ${formatFileSize(e.total)}`;
            }
        });
        
        // Handle upload completion
        xhr.addEventListener('load', function() {
            if (xhr.status >= 200 && xhr.status < 300) {
                const response = JSON.parse(xhr.responseText);
                
                if (response.success) {
                    // Store dataset ID
                    currentDatasetId = response.dataset_id;
                    
                    // Show success message
                    showUploadSuccess('File uploaded successfully!');
                    
                    // Display dataset information
                    displayDatasetInfo(response);
                } else {
                    showUploadError(response.error || 'Upload failed');
                }
            } else {
                try {
                    const response = JSON.parse(xhr.responseText);
                    showUploadError(response.error || 'Upload failed');
                } catch (e) {
                    showUploadError('Upload failed: ' + xhr.statusText);
                }
            }
            
            // Reset upload UI
            resetUploadUI();
        });
        
        // Handle upload error
        xhr.addEventListener('error', function() {
            showUploadError('Network error occurred during upload');
            resetUploadUI();
        });
        
        // Handle upload abort
        xhr.addEventListener('abort', function() {
            showUploadError('Upload was aborted');
            resetUploadUI();
        });
        
        // Send the request
        xhr.open('POST', '/upload', true);
        xhr.send(formData);
        
        uploadStatus.textContent = 'Preparing upload...';
    }
    
    function resetUploadUI() {
        uploadPrompt.classList.remove('d-none');
        uploadProgress.classList.add('d-none');
        progressBar.style.width = '0%';
        progressBar.textContent = '';
        fileInput.value = '';
    }
    
    function showUploadError(message) {
        uploadError.textContent = message;
        uploadError.classList.remove('d-none');
        uploadSuccess.classList.add('d-none');
    }
    
    function showUploadSuccess(message) {
        uploadSuccess.textContent = message;
        uploadSuccess.classList.remove('d-none');
        uploadError.classList.add('d-none');
    }
    
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    function displayDatasetInfo(data) {
        // Show dataset info section
        document.getElementById('datasetInfo').classList.remove('d-none');
        
        // Update stats
        document.getElementById('datasetTotal').textContent = data.stats.total;
        document.getElementById('datasetNormal').textContent = data.stats.non_fraud;
        document.getElementById('datasetFraud').textContent = data.stats.fraud;
        document.getElementById('datasetFraudPercentage').textContent = data.stats.fraud_percentage;
        
        // Update preview table
        const previewTableBody = document.getElementById('previewTableBody');
        previewTableBody.innerHTML = '';
        
        data.preview.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${row.Time}</td>
                <td>$${row.Amount.toFixed(2)}</td>
                <td>${row.Class === 1 ? 'Fraud' : 'Normal'}</td>
            `;
            previewTableBody.appendChild(tr);
        });
    }
}

function analyzeDataset(datasetId) {
    // Show loading state
    document.getElementById('analyzeDatasetBtn').disabled = true;
    document.getElementById('analyzeDatasetBtn').innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...';
    
    // Fetch analysis results
    fetch(`/dataset/${datasetId}/analyze`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Show analysis results section
                document.getElementById('analysisResultsSection').classList.remove('d-none');
                
                // Update summary stats
                document.getElementById('analysisTotalRecords').textContent = data.total_records;
                document.getElementById('analysisAnomaliesDetected').textContent = data.anomalies_detected;
                document.getElementById('analysisAnomalyPercentage').textContent = data.anomalies_percentage;
                
                // Update accuracy if available
                if (data.accuracy !== null) {
                    document.getElementById('analysisAccuracy').textContent = data.accuracy + '%';
                } else {
                    document.getElementById('analysisAccuracy').textContent = 'N/A (No class labels)';
                }
                
                // Update confusion matrix if available
                if (data.confusion_matrix) {
                    document.getElementById('confusionMatrixCard').classList.remove('d-none');
                    document.getElementById('cmTP').textContent = `TP: ${data.confusion_matrix.true_positive}`;
                    document.getElementById('cmFP').textContent = `FP: ${data.confusion_matrix.false_positive}`;
                    document.getElementById('cmFN').textContent = `FN: ${data.confusion_matrix.false_negative}`;
                    document.getElementById('cmTN').textContent = `TN: ${data.confusion_matrix.true_negative}`;
                } else {
                    document.getElementById('confusionMatrixCard').classList.add('d-none');
                }
                
                // Update results preview table
                const analysisTableBody = document.getElementById('analysisTableBody');
                analysisTableBody.innerHTML = '';
                
                data.preview_with_predictions.forEach(row => {
                    const tr = document.createElement('tr');
                    tr.className = row.Prediction !== row.Class ? 'table-warning' : '';
                    tr.innerHTML = `
                        <td>${row.Time}</td>
                        <td>$${row.Amount.toFixed(2)}</td>
                        <td>${row.Class === 1 ? 'Fraud' : 'Normal'}</td>
                        <td>${row.Prediction === 1 ? 'Fraud' : 'Normal'}</td>
                    `;
                    analysisTableBody.appendChild(tr);
                });
            } else {
                alert('Error analyzing dataset: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while analyzing the dataset.');
        })
        .finally(() => {
            // Reset button state
            document.getElementById('analyzeDatasetBtn').disabled = false;
            document.getElementById('analyzeDatasetBtn').textContent = 'Analyze Dataset';
        });
}

function clearDataset(datasetId) {
    if (confirm('Are you sure you want to remove this dataset?')) {
        fetch(`/dataset/${datasetId}`, {
            method: 'DELETE'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Reset UI
                document.getElementById('datasetInfo').classList.add('d-none');
                document.getElementById('analysisResultsSection').classList.add('d-none');
                currentDatasetId = null;
                
                // Show success message
                const uploadSuccess = document.getElementById('uploadSuccess');
                uploadSuccess.textContent = 'Dataset removed successfully';
                uploadSuccess.classList.remove('d-none');
                document.getElementById('uploadError').classList.add('d-none');
            } else {
                alert('Error removing dataset: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while removing the dataset.');
        });
    }
}

// Pagination variables
let currentPage = 1;
let itemsPerPage = 10;
let classFilter = 'all';
let totalPages = 0;
let totalRecords = 0;

function initPagination() {
    // Load initial data
    loadPaginatedData();
    
    // Set up event listeners for pagination controls
    document.getElementById('perPageSelect').addEventListener('change', function() {
        itemsPerPage = parseInt(this.value);
        currentPage = 1; // Reset to first page when changing items per page
        loadPaginatedData();
    });
    
    document.getElementById('classFilter').addEventListener('change', function() {
        classFilter = this.value;
        currentPage = 1; // Reset to first page when changing filter
        loadPaginatedData();
    });
}

function loadPaginatedData() {
    // Show loading indicator
    const loadingIndicator = document.getElementById('loadingIndicator');
    const noDataMessage = document.getElementById('noDataMessage');
    const tableBody = document.getElementById('paginatedTableBody');
    
    loadingIndicator.classList.remove('d-none');
    noDataMessage.classList.add('d-none');
    tableBody.innerHTML = '';
    
    // Prepare URL with query parameters
    let url = `/filter_data?page=${currentPage}&per_page=${itemsPerPage}`;
    
    if (classFilter !== 'all') {
        url += `&class=${classFilter}`;
    }
    
    // Fetch paginated data
    fetch(url)
        .then(response => response.json())
        .then(data => {
            // Hide loading indicator
            loadingIndicator.classList.add('d-none');
            
            // Update pagination metadata
            totalPages = data.pagination.total_pages;
            totalRecords = data.pagination.total_records;
            
            // Update pagination info text
            document.getElementById('showingFrom').textContent = data.data.length > 0 ? 
                (data.pagination.page - 1) * data.pagination.per_page + 1 : 0;
            document.getElementById('showingTo').textContent = data.data.length > 0 ? 
                (data.pagination.page - 1) * data.pagination.per_page + data.data.length : 0;
            document.getElementById('totalRecords').textContent = totalRecords;
            
            // Render table rows
            if (data.data.length > 0) {
                data.data.forEach(row => {
                    const tr = document.createElement('tr');
                    tr.className = row.Class === 1 ? 'table-danger' : '';
                    
                    tr.innerHTML = `
                        <td>${row.index}</td>
                        <td>${row.Time}</td>
                        <td>$${row.Amount.toFixed(2)}</td>
                        <td>${row.V1.toFixed(4)}</td>
                        <td>${row.V2.toFixed(4)}</td>
                        <td>${row.V3.toFixed(4)}</td>
                        <td>${row.V4.toFixed(4)}</td>
                        <td>${row.Class === 1 ? 'Fraud' : 'Normal'}</td>
                        <td>
                            <button class="btn btn-sm btn-primary paginated-detect-btn" data-index="${row.index}">Detect</button>
                        </td>
                    `;
                    
                    tableBody.appendChild(tr);
                });
                
                // Add event listeners to the new detect buttons
                document.querySelectorAll('.paginated-detect-btn').forEach(button => {
                    button.addEventListener('click', function() {
                        const rowIndex = this.getAttribute('data-index');
                        detectAnomaly(rowIndex);
                    });
                });
            } else {
                noDataMessage.classList.remove('d-none');
            }
            
            // Generate pagination controls
            generatePaginationControls();
        })
        .catch(error => {
            console.error('Error loading paginated data:', error);
            loadingIndicator.classList.add('d-none');
            noDataMessage.classList.remove('d-none');
            noDataMessage.textContent = 'Error loading data. Please try again.';
        });
}

function generatePaginationControls() {
    const paginationControls = document.getElementById('paginationControls');
    paginationControls.innerHTML = '';
    
    // Previous button
    const prevLi = document.createElement('li');
    prevLi.className = `page-item ${currentPage === 1 ? 'disabled' : ''}`;
    prevLi.innerHTML = `<a class="page-link" href="#" aria-label="Previous"><span aria-hidden="true">&laquo;</span></a>`;
    prevLi.addEventListener('click', function(e) {
        e.preventDefault();
        if (currentPage > 1) {
            currentPage--;
            loadPaginatedData();
        }
    });
    paginationControls.appendChild(prevLi);
    
    // Page numbers
    const maxVisiblePages = 5;
    let startPage = Math.max(1, currentPage - Math.floor(maxVisiblePages / 2));
    let endPage = Math.min(totalPages, startPage + maxVisiblePages - 1);
    
    // Adjust start page if we're near the end
    if (endPage - startPage + 1 < maxVisiblePages && startPage > 1) {
        startPage = Math.max(1, endPage - maxVisiblePages + 1);
    }
    
    // First page button if not visible
    if (startPage > 1) {
        const firstLi = document.createElement('li');
        firstLi.className = 'page-item';
        firstLi.innerHTML = `<a class="page-link" href="#">1</a>`;
        firstLi.addEventListener('click', function(e) {
            e.preventDefault();
            currentPage = 1;
            loadPaginatedData();
        });
        paginationControls.appendChild(firstLi);
        
        // Ellipsis if needed
        if (startPage > 2) {
            const ellipsisLi = document.createElement('li');
            ellipsisLi.className = 'page-item disabled';
            ellipsisLi.innerHTML = `<a class="page-link" href="#">...</a>`;
            paginationControls.appendChild(ellipsisLi);
        }
    }
    
    // Page number buttons
    for (let i = startPage; i <= endPage; i++) {
        const pageLi = document.createElement('li');
        pageLi.className = `page-item ${i === currentPage ? 'active' : ''}`;
        pageLi.innerHTML = `<a class="page-link" href="#">${i}</a>`;
        pageLi.addEventListener('click', function(e) {
            e.preventDefault();
            currentPage = i;
            loadPaginatedData();
        });
        paginationControls.appendChild(pageLi);
    }
    
    // Ellipsis and last page if needed
    if (endPage < totalPages) {
        if (endPage < totalPages - 1) {
            const ellipsisLi = document.createElement('li');
            ellipsisLi.className = 'page-item disabled';
            ellipsisLi.innerHTML = `<a class="page-link" href="#">...</a>`;
            paginationControls.appendChild(ellipsisLi);
        }
        
        const lastLi = document.createElement('li');
        lastLi.className = 'page-item';
        lastLi.innerHTML = `<a class="page-link" href="#">${totalPages}</a>`;
        lastLi.addEventListener('click', function(e) {
            e.preventDefault();
            currentPage = totalPages;
            loadPaginatedData();
        });
        paginationControls.appendChild(lastLi);
    }
    
    // Next button
    const nextLi = document.createElement('li');
    nextLi.className = `page-item ${currentPage === totalPages ? 'disabled' : ''}`;
    nextLi.innerHTML = `<a class="page-link" href="#" aria-label="Next"><span aria-hidden="true">&raquo;</span></a>`;
    nextLi.addEventListener('click', function(e) {
        e.preventDefault();
        if (currentPage < totalPages) {
            currentPage++;
            loadPaginatedData();
        }
    });
    paginationControls.appendChild(nextLi);
}

function fetchDataAndRenderCharts() {
    fetch('/data')
        .then(response => response.json())
        .then(data => {
            renderAmountChart(data.amount_stats);
            renderCorrelationChart(data.correlations.top_positive, data.correlations.top_negative);
        })
        .catch(error => console.error('Error fetching data:', error));
}

function renderAmountChart(amountStats) {
    const ctx = document.getElementById('amountChart').getContext('2d');
    
    const chartData = {
        labels: ['Mean', 'Median', 'Max'],
        datasets: [
            {
                label: 'Normal Transactions',
                backgroundColor: 'rgba(40, 167, 69, 0.7)',
                data: [
                    amountStats.non_fraud.mean,
                    amountStats.non_fraud.median,
                    amountStats.non_fraud.max
                ]
            },
            {
                label: 'Fraudulent Transactions',
                backgroundColor: 'rgba(220, 53, 69, 0.7)',
                data: [
                    amountStats.fraud.mean,
                    amountStats.fraud.median,
                    amountStats.fraud.max
                ]
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

function renderCorrelationChart(positiveCorrelations, negativeCorrelations) {
    const ctx = document.getElementById('correlationChart').getContext('2d');
    
    // Combine top positive and negative correlations
    const topPositive = Object.entries(positiveCorrelations)
        .filter(([key]) => key !== 'Class') // Exclude Class itself
        .slice(0, 5); // Take top 5 positive
    
    const topNegative = Object.entries(negativeCorrelations)
        .filter(([key]) => key !== 'Class') // Exclude Class itself
        .slice(0, 5); // Take top 5 negative
    
    const combined = [...topPositive, ...topNegative];
    
    const labels = combined.map(item => item[0]);
    const values = combined.map(item => item[1]);
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
                    text: 'Top Features Correlated with Fraud'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Correlation: ${context.raw.toFixed(4)}`;
                        }
                    }
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