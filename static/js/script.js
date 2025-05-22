// Variables
let image1 = null;
let image2 = null;
let isProcessing = false;
let currentResultId = null;

// DOM Elements
const elements = {
    // Upload elements
    load1Button: document.getElementById('load1'),
    load2Button: document.getElementById('load2'),
    file1Input: document.getElementById('file1'),
    file2Input: document.getElementById('file2'),
    preview1: document.getElementById('preview1'),
    preview2: document.getElementById('preview2'),
    info1: document.getElementById('info1'),
    info2: document.getElementById('info2'),
    
    // Configuration elements
    modelSelect: document.getElementById('model-select'),
    detectorSelect: document.getElementById('detector-select'),
    distanceSelect: document.getElementById('distance-select'),
    
    // Control elements
    predictButton: document.getElementById('predict'),
    loadingElement: document.getElementById('loading'),
    errorElement: document.getElementById('error'),
    errorText: document.getElementById('error-text'),
    
    // Results elements
    resultsSection: document.getElementById('results-section'),
    similarityPercentage: document.getElementById('similarity-percentage'),
    similarityStatus: document.getElementById('similarity-status'),
    similarityBar: document.getElementById('similarity-bar'),
    analysisDetails: document.getElementById('analysis-details'),
    
    // Face attributes
    face1Age: document.getElementById('face1-age'),
    face1Gender: document.getElementById('face1-gender'),
    face1Race: document.getElementById('face1-race'),
    face1Emotion: document.getElementById('face1-emotion'),
    face1Type: document.getElementById('face1-type'),
    face2Age: document.getElementById('face2-age'),
    face2Gender: document.getElementById('face2-gender'),
    face2Race: document.getElementById('face2-race'),
    face2Emotion: document.getElementById('face2-emotion'),
    face2Type: document.getElementById('face2-type'),
    
    // Share elements
    shareTwitter: document.getElementById('share-twitter'),
    shareFacebook: document.getElementById('share-facebook'),
    shareLinkedin: document.getElementById('share-linkedin'),
    copyLink: document.getElementById('copy-link'),
    
    // Status elements
    systemStatus: document.getElementById('system-status'),
    modelStatus: document.getElementById('model-status'),
    statusIndicator: document.getElementById('status-indicator'),
    
    // Loading steps
    step1: document.getElementById('step1'),
    step2: document.getElementById('step2'),
    step3: document.getElementById('step3'),
    step4: document.getElementById('step4')
};

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    checkSystemStatus();
    loadAvailableModels();
});

function initializeEventListeners() {
    // Upload buttons
    elements.load1Button.addEventListener('click', () => elements.file1Input.click());
    elements.load2Button.addEventListener('click', () => elements.file2Input.click());
    
    // File inputs
    elements.file1Input.addEventListener('change', (e) => handleImageUpload(e, 1));
    elements.file2Input.addEventListener('change', (e) => handleImageUpload(e, 2));
    
    // Preview click to upload
    elements.preview1.addEventListener('click', () => elements.file1Input.click());
    elements.preview2.addEventListener('click', () => elements.file2Input.click());
    
    // Predict button
    elements.predictButton.addEventListener('click', predictSimilarity);
    
    // Share buttons
    elements.shareTwitter.addEventListener('click', shareToTwitter);
    elements.shareFacebook.addEventListener('click', shareToFacebook);
    elements.shareLinkedin.addEventListener('click', shareToLinkedin);
    elements.copyLink.addEventListener('click', copyShareLink);
}

function handleImageUpload(event, imageNumber) {
    const file = event.target.files[0];
    if (!file) return;
    
    hideError();
    
    // Validate file
    if (!file.type.match('image.*')) {
        showError('Please select an image file (JPG, PNG, WEBP, etc.)');
        return;
    }
    
    if (file.size > 15 * 1024 * 1024) { // 15MB limit
        showError('Image is too large. Please select an image under 15MB');
        return;
    }
    
    const previewElement = imageNumber === 1 ? elements.preview1 : elements.preview2;
    const infoElement = imageNumber === 1 ? elements.info1 : elements.info2;
    
    // Show loading
    previewElement.innerHTML = '<div class="preview-placeholder"><i class="fas fa-spinner fa-spin"></i><p>Loading image...</p></div>';
    
    const reader = new FileReader();
    reader.onload = function(e) {
        // Create image element
        const img = document.createElement('img');
        img.src = e.target.result;
        
        img.onload = function() {
            // Clear preview and add image
            previewElement.innerHTML = '';
            previewElement.appendChild(img);
            
            // Store image data
            if (imageNumber === 1) {
                image1 = e.target.result;
            } else {
                image2 = e.target.result;
            }
            
            // Show image info
            showImageInfo(file, infoElement);
            updatePredictButtonState();
        };
        
        img.onerror = function() {
            showImageUploadError(previewElement, 'Failed to load image');
            if (imageNumber === 1) image1 = null;
            else image2 = null;
            updatePredictButtonState();
        };
    };
    
    reader.onerror = function() {
        showImageUploadError(previewElement, 'Failed to read file');
    };
    
    reader.readAsDataURL(file);
}

function showImageInfo(file, infoElement) {
    const fileSize = (file.size / 1024 / 1024).toFixed(2);
    infoElement.innerHTML = `
        <strong>File:</strong> ${file.name}<br>
        <strong>Size:</strong> ${fileSize} MB<br>
        <strong>Type:</strong> ${file.type}
    `;
    infoElement.style.display = 'block';
}

function showImageUploadError(previewElement, message) {
    previewElement.innerHTML = `
        <div class="preview-placeholder">
            <i class="fas fa-exclamation-triangle" style="color: #e74c3c;"></i>
            <p style="color: #e74c3c;">${message}</p>
        </div>
    `;
}

function updatePredictButtonState() {
    const canPredict = image1 && image2 && !isProcessing;
    elements.predictButton.disabled = !canPredict;
    
    if (isProcessing) {
        elements.predictButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> <span>Processing...</span>';
    } else {
        elements.predictButton.innerHTML = '<i class="fas fa-search"></i> <span>Analyze Faces</span>';
    }
}

function predictSimilarity() {
    if (!image1 || !image2) {
        showError('Please upload both images first');
        return;
    }
    
    isProcessing = true;
    updatePredictButtonState();
    
    // Hide previous results and errors
    hideError();
    elements.resultsSection.style.display = 'none';
    
    // Show loading with steps
    showLoadingSteps();
    
    // Prepare request data
    const requestData = {
        image1: image1,
        image2: image2,
        model: elements.modelSelect.value,
        detector: elements.detectorSelect.value,
        distance_metric: elements.distanceSelect.value
    };
    
    // Send request
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'Analysis failed');
            });
        }
        return response.json();
    })
    .then(data => {
        displayResults(data);
        currentResultId = data.result_id;
    })
    .catch(error => {
        console.error('Error:', error);
        showError(error.message || 'Failed to analyze images. Please try again.');
    })
    .finally(() => {
        hideLoading();
        isProcessing = false;
        updatePredictButtonState();
    });
}

function showLoadingSteps() {
    elements.loadingElement.style.display = 'block';
    
    // Animate steps
    const steps = [elements.step1, elements.step2, elements.step3, elements.step4];
    steps.forEach(step => step.classList.remove('active'));
    
    let currentStep = 0;
    const stepInterval = setInterval(() => {
        if (currentStep < steps.length) {
            steps[currentStep].classList.add('active');
            currentStep++;
        } else {
            clearInterval(stepInterval);
        }
    }, 500);
}

function hideLoading() {
    elements.loadingElement.style.display = 'none';
}

function displayResults(data) {
    // Show results section
    elements.resultsSection.style.display = 'block';
    
    // Display similarity score
    elements.similarityPercentage.textContent = `${data.similarity.toFixed(1)}%`;
    elements.similarityBar.style.width = `${data.similarity}%`;
    
    // Update similarity status
    let status = '';
    let statusColor = '';
    
    if (data.verified || data.similarity > 80) {
        status = 'Very High Similarity - Same Person';
        statusColor = '#27ae60';
    } else if (data.similarity > 60) {
        status = 'High Similarity - Likely Same Person';
        statusColor = '#2ecc71';
    } else if (data.similarity > 40) {
        status = 'Moderate Similarity - Possibly Same Person';
        statusColor = '#f39c12';
    } else if (data.similarity > 20) {
        status = 'Low Similarity - Different People';
        statusColor = '#e67e22';
    } else {
        status = 'Very Low Similarity - Different People';
        statusColor = '#e74c3c';
    }
    
    elements.similarityStatus.textContent = status;
    elements.similarityPercentage.style.color = statusColor;
    elements.similarityBar.style.background = `linear-gradient(135deg, ${statusColor} 0%, ${statusColor}dd 100%)`;
    
    // Display analysis details
    elements.analysisDetails.innerHTML = `
        <strong>Analysis Method:</strong> ${data.method_used}<br>
        <strong>Model:</strong> ${data.model_name}<br>
        <strong>Detector:</strong> ${data.detector_backend}<br>
        <strong>Distance Metric:</strong> ${data.distance_metric}<br>
        <strong>Distance Score:</strong> ${data.distance.toFixed(4)}<br>
        <strong>Threshold:</strong> ${data.threshold.toFixed(4)}
    `;
    
    // Display face attributes
    displayFaceAttributes(data.face1_attributes, 1);
    displayFaceAttributes(data.face2_attributes, 2);
    
    // Display image types
    elements.face1Type.textContent = data.img1_type;
    elements.face2Type.textContent = data.img2_type;
    
    // Scroll to results
    elements.resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function displayFaceAttributes(attributes, faceNumber) {
    if (!attributes) {
        // Set default values if attributes are not available
        const defaultText = 'Not detected';
        if (faceNumber === 1) {
            elements.face1Age.textContent = defaultText;
            elements.face1Gender.textContent = defaultText;
            elements.face1Race.textContent = defaultText;
            elements.face1Emotion.textContent = defaultText;
        } else {
            elements.face2Age.textContent = defaultText;
            elements.face2Gender.textContent = defaultText;
            elements.face2Race.textContent = defaultText;
            elements.face2Emotion.textContent = defaultText;
        }
        return;
    }
    
    // Update face attributes
    if (faceNumber === 1) {
        elements.face1Age.textContent = `${attributes.age} years`;
        elements.face1Gender.textContent = `${attributes.gender} (${attributes.gender_confidence.toFixed(1)}%)`;
        elements.face1Race.textContent = `${attributes.race} (${attributes.race_confidence.toFixed(1)}%)`;
        elements.face1Emotion.textContent = `${attributes.emotion} (${attributes.emotion_confidence.toFixed(1)}%)`;
    } else {
        elements.face2Age.textContent = `${attributes.age} years`;
        elements.face2Gender.textContent = `${attributes.gender} (${attributes.gender_confidence.toFixed(1)}%)`;
        elements.face2Race.textContent = `${attributes.race} (${attributes.race_confidence.toFixed(1)}%)`;
        elements.face2Emotion.textContent = `${attributes.emotion} (${attributes.emotion_confidence.toFixed(1)}%)`;
    }
}

// Share functions
function shareToTwitter() {
    if (!currentResultId) return;
    
    const text = `Check out my AI face analysis results! Similarity: ${elements.similarityPercentage.textContent}`;
    const url = `${window.location.origin}/share/${currentResultId}`;
    const twitterUrl = `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(url)}`;
    
    window.open(twitterUrl, '_blank');
}

function shareToFacebook() {
    if (!currentResultId) return;
    
    const url = `${window.location.origin}/share/${currentResultId}`;
    const facebookUrl = `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(url)}`;
    
    window.open(facebookUrl, '_blank');
}

function shareToLinkedin() {
    if (!currentResultId) return;
    
    const url = `${window.location.origin}/share/${currentResultId}`;
    const title = 'AI Face Analysis Results';
    const linkedinUrl = `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(url)}`;
    
    window.open(linkedinUrl, '_blank');
}

function copyShareLink() {
    if (!currentResultId) return;
    
    const url = `${window.location.origin}/share/${currentResultId}`;
    
    navigator.clipboard.writeText(url).then(() => {
        // Show success feedback
        const originalText = elements.copyLink.innerHTML;
        elements.copyLink.innerHTML = '<i class="fas fa-check"></i> Copied!';
        elements.copyLink.style.background = '#28a745';
        
        setTimeout(() => {
            elements.copyLink.innerHTML = originalText;
            elements.copyLink.style.background = '#6c757d';
        }, 2000);
    }).catch(() => {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = url;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        
        showError('Link copied to clipboard!');
        setTimeout(hideError, 3000);
    });
}

// System status functions
function checkSystemStatus() {
    elements.systemStatus.textContent = 'Checking system...';
    elements.statusIndicator.className = 'fas fa-circle';
    
    fetch('/test_deepface')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                elements.systemStatus.textContent = 'System Online';
                elements.statusIndicator.className = 'fas fa-circle online';
                elements.modelStatus.textContent = `${data.available_models.length} models available`;
            } else {
                elements.systemStatus.textContent = 'Limited Functionality';
                elements.statusIndicator.className = 'fas fa-circle offline';
                elements.modelStatus.textContent = 'Some features may be unavailable';
            }
        })
        .catch(error => {
            elements.systemStatus.textContent = 'System Offline';
            elements.statusIndicator.className = 'fas fa-circle offline';
            elements.modelStatus.textContent = 'Connection failed';
        });
}

function loadAvailableModels() {
    fetch('/get_models')
        .then(response => response.json())
        .then(data => {
            // Update model options
            elements.modelSelect.innerHTML = '';
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                if (model === 'Facenet') option.selected = true;
                elements.modelSelect.appendChild(option);
            });
            
            // Update detector options
            elements.detectorSelect.innerHTML = '';
            data.detectors.forEach(detector => {
                const option = document.createElement('option');
                option.value = detector;
                option.textContent = detector.charAt(0).toUpperCase() + detector.slice(1);
                if (detector === 'opencv') option.selected = true;
                elements.detectorSelect.appendChild(option);
            });
            
            // Update distance metric options
            elements.distanceSelect.innerHTML = '';
            data.distance_metrics.forEach(metric => {
                const option = document.createElement('option');
                option.value = metric;
                option.textContent = metric.charAt(0).toUpperCase() + metric.slice(1);
                if (metric === 'cosine') option.selected = true;
                elements.distanceSelect.appendChild(option);
            });
        })
        .catch(error => {
            console.error('Failed to load available models:', error);
        });
}

// Utility functions
function showError(message) {
    elements.errorText.textContent = message;
    elements.errorElement.style.display = 'flex';
    
    // Auto-hide after 8 seconds
    setTimeout(hideError, 8000);
}

function hideError() {
    elements.errorElement.style.display = 'none';
}

// Drag and drop functionality
function setupDragAndDrop() {
    [elements.preview1, elements.preview2].forEach((preview, index) => {
        preview.addEventListener('dragover', (e) => {
            e.preventDefault();
            preview.style.borderColor = '#667eea';
            preview.style.background = 'rgba(102, 126, 234, 0.1)';
        });
        
        preview.addEventListener('dragleave', (e) => {
            e.preventDefault();
            preview.style.borderColor = '#e1e8ed';
            preview.style.background = '';
        });
        
        preview.addEventListener('drop', (e) => {
            e.preventDefault();
            preview.style.borderColor = '#e1e8ed';
            preview.style.background = '';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (file.type.match('image.*')) {
                    // Create a fake event object
                    const fakeEvent = {
                        target: {
                            files: [file]
                        }
                    };
                    handleImageUpload(fakeEvent, index + 1);
                }
            }
        });
    });
}

// Enhanced image analysis
function analyzeImageQuality(file, callback) {
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    img.onload = function() {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        
        // Calculate image quality metrics
        let brightness = 0;
        let contrast = 0;
        
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            const gray = (r + g + b) / 3;
            brightness += gray;
        }
        
        brightness = brightness / (data.length / 4);
        
        // Simple quality assessment
        const quality = {
            brightness: brightness,
            resolution: img.width * img.height,
            aspectRatio: img.width / img.height,
            fileSize: file.size
        };
        
        callback(quality);
    };
    
    img.src = URL.createObjectURL(file);
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to analyze
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (!isProcessing && image1 && image2) {
            predictSimilarity();
        }
    }
    
    // Escape to close results
    if (e.key === 'Escape') {
        elements.resultsSection.style.display = 'none';
    }
});

// Initialize drag and drop
setupDragAndDrop();

// Performance monitoring
let analysisStartTime;
let performanceMetrics = [];

function startPerformanceMonitoring() {
    analysisStartTime = performance.now();
}

function endPerformanceMonitoring() {
    if (analysisStartTime) {
        const duration = performance.now() - analysisStartTime;
        performanceMetrics.push({
            timestamp: new Date().toISOString(),
            duration: duration,
            model: elements.modelSelect.value,
            detector: elements.detectorSelect.value
        });
        
        console.log(`Analysis completed in ${duration.toFixed(2)}ms`);
        
        // Keep only last 10 measurements
        if (performanceMetrics.length > 10) {
            performanceMetrics.shift();
        }
    }
}

// Enhanced predict function with performance monitoring
const originalPredictSimilarity = predictSimilarity;
predictSimilarity = function() {
    startPerformanceMonitoring();
    return originalPredictSimilarity.apply(this, arguments);
};

// Override displayResults to include performance monitoring
const originalDisplayResults = displayResults;
displayResults = function(data) {
    endPerformanceMonitoring();
    return originalDisplayResults.apply(this, arguments);
};

// Auto-save user preferences
function saveUserPreferences() {
    const preferences = {
        model: elements.modelSelect.value,
        detector: elements.detectorSelect.value,
        distanceMetric: elements.distanceSelect.value
    };
    
    localStorage.setItem('faceAnalysisPreferences', JSON.stringify(preferences));
}

function loadUserPreferences() {
    const saved = localStorage.getItem('faceAnalysisPreferences');
    if (saved) {
        try {
            const preferences = JSON.parse(saved);
            if (preferences.model) elements.modelSelect.value = preferences.model;
            if (preferences.detector) elements.detectorSelect.value = preferences.detector;
            if (preferences.distanceMetric) elements.distanceSelect.value = preferences.distanceMetric;
        } catch (e) {
            console.error('Error loading preferences:', e);
        }
    }
}

// Save preferences when changed
elements.modelSelect.addEventListener('change', saveUserPreferences);
elements.detectorSelect.addEventListener('change', saveUserPreferences);
elements.distanceSelect.addEventListener('change', saveUserPreferences);

// Load preferences on page load
setTimeout(loadUserPreferences, 1000); // Wait for models to load

// Advanced error handling
window.addEventListener('error', function(e) {
    console.error('JavaScript error:', e.error);
    showError('An unexpected error occurred. Please refresh the page and try again.');
});

window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
    showError('A network error occurred. Please check your connection and try again.');
});

console.log('AI Face Analysis Pro - Advanced Features Loaded');
console.log('Keyboard shortcuts: Ctrl+Enter to analyze, Esc to close results');
console.log('Drag and drop images directly onto the preview areas');