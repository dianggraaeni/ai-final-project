// Variables to store the image data
let image1 = null;
let image2 = null;
let isProcessing = false;

// DOM Elements
const load1Button = document.getElementById('load1');
const load2Button = document.getElementById('load2');
const file1Input = document.getElementById('file1');
const file2Input = document.getElementById('file2');
const preview1 = document.getElementById('preview1');
const preview2 = document.getElementById('preview2');
const predictButton = document.getElementById('predict');
const resultElement = document.getElementById('result');
const resultDetailsElement = document.getElementById('result-details');
const loadingElement = document.getElementById('loading');
const errorElement = document.getElementById('error');
const systemStatusElement = document.getElementById('system-status');

// Check if backend is available
checkSystemStatus();

// Event Listeners
load1Button.addEventListener('click', () => {
    file1Input.click();
});

load2Button.addEventListener('click', () => {
    file2Input.click();
});

file1Input.addEventListener('change', (event) => {
    handleImageUpload(event, preview1, 1);
});

file2Input.addEventListener('change', (event) => {
    handleImageUpload(event, preview2, 2);
});

predictButton.addEventListener('click', predictSimilarity);

// Functions
function handleImageUpload(event, previewElement, imageNumber) {
    const file = event.target.files[0];
    if (!file) return;
    
    // Clear previous error messages
    hideError();
    
    // Validate file type
    if (!file.type.match('image.*')) {
        showError('Please select an image file (JPEG, PNG, etc.)');
        return;
    }
    
    // More lenient file size check - allow up to 10MB
    if (file.size > 10 * 1024 * 1024) { 
        showError('Image is too large. Please select an image under 10MB');
        return;
    }

    const reader = new FileReader();
    
    // Show loading in preview
    previewElement.innerHTML = '<p>Loading image...</p>';
    
    reader.onload = function(e) {
        // Clear previous content
        previewElement.innerHTML = '';
        
        // Create image element
        const img = document.createElement('img');
        img.src = e.target.result;
        
        // Store image data
        if (imageNumber === 1) {
            image1 = e.target.result;
        } else {
            image2 = e.target.result;
        }
        
        // Add image to preview when loaded
        img.onload = function() {
            previewElement.appendChild(img);
            updatePredictButtonState();
        };
        
        img.onerror = function() {
            previewElement.innerHTML = '<p>Failed to load image</p>';
            showError('The selected file could not be loaded as an image');
            if (imageNumber === 1) {
                image1 = null;
            } else {
                image2 = null;
            }
            updatePredictButtonState();
        };
    };
    
    reader.onerror = function() {
        previewElement.innerHTML = '<p>Failed to read file</p>';
        showError('Failed to read the selected file');
    };
    
    reader.readAsDataURL(file);
}

function updatePredictButtonState() {
    // Enable predict button only if both images are loaded
    if (image1 && image2 && !isProcessing) {
        predictButton.disabled = false;
        predictButton.textContent = 'Predict Similarity';
    } else {
        predictButton.disabled = true;
        if (isProcessing) {
            predictButton.textContent = 'Processing...';
        } else {
            predictButton.textContent = 'Predict Similarity';
        }
    }
}

function predictSimilarity() {
    if (!image1 || !image2) {
        showError('Please upload both images first');
        return;
    }
    
    // Disable predict button during processing
    isProcessing = true;
    updatePredictButtonState();
    
    // Show loading indicator
    loadingElement.style.display = 'block';
    resultElement.textContent = 'Processing...';
    resultDetailsElement.textContent = '';
    hideError();
    
    // Send images to backend API with extended timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout
    
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            image1: image1,
            image2: image2
        }),
        signal: controller.signal
    })
    .then(response => {
        clearTimeout(timeoutId);
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'An error occurred during prediction');
            });
        }
        return response.json();
    })
    .then(data => {
        // Format and display results
        resultElement.textContent = `${data.similarity.toFixed(2)}% Similar`;
        
        // Add method information
        let methodInfo = '';
        if (data.method === 'histogram') {
            methodInfo = ' (using histogram comparison)';
        } else if (data.model_used) {
            methodInfo = ` (using ${data.model_used} model)`;
        }
        
        // Update message based on similarity level
        if (data.verified || data.similarity > 70) {
            resultDetailsElement.textContent = `High similarity detected. These appear to be the same person.${methodInfo}`;
            resultElement.style.color = '#27ae60';
        } else if (data.similarity > 50) {
            resultDetailsElement.textContent = `Moderate similarity detected. These might be the same person.${methodInfo}`;
            resultElement.style.color = '#f39c12';
        } else if (data.similarity > 30) {
            resultDetailsElement.textContent = `Low similarity detected. These are likely different people.${methodInfo}`;
            resultElement.style.color = '#e67e22';
        } else {
            resultDetailsElement.textContent = `Very low similarity detected. These are different people.${methodInfo}`;
            resultElement.style.color = '#e74c3c';
        }
    })
    .catch(error => {
        clearTimeout(timeoutId);
        console.error('Error:', error);
        
        if (error.name === 'AbortError') {
            showError('Request timed out. Please try again with smaller images.');
        } else {
            showError(error.message || 'Failed to analyze images. Please try again.');
        }
        
        resultElement.textContent = 'Analysis failed';
        resultDetailsElement.textContent = '';
    })
    .finally(() => {
        loadingElement.style.display = 'none';
        isProcessing = false;
        updatePredictButtonState();
    });
}

function showError(message) {
    errorElement.textContent = message;
    errorElement.style.display = 'block';
    
    // Auto-hide error after 10 seconds
    setTimeout(() => {
        hideError();
    }, 10000);
}

function hideError() {
    errorElement.style.display = 'none';
}

function checkSystemStatus() {
    systemStatusElement.textContent = 'Checking...';
    
    fetch('/test_deepface')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                systemStatusElement.textContent = 'Online';
                systemStatusElement.className = 'online';
            } else {
                systemStatusElement.textContent = 'Limited Functionality';
                systemStatusElement.className = 'offline';
                console.warn('System issue detected:', data.message);
            }
        })
        .catch(error => {
            systemStatusElement.textContent = 'Basic Mode';
            systemStatusElement.className = 'offline';
            console.error('Status check failed:', error);
        });
}

// Initialize UI state
updatePredictButtonState();