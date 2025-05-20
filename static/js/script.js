// Variables to store the image data
let image1 = null;
let image2 = null;

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
    
    if (!file.type.match('image.*')) {
        showError('Please select an image file');
        return;
    }

    const reader = new FileReader();
    
    reader.onload = function(e) {
        // Clear previous content
        previewElement.innerHTML = '';
        
        // Create image element
        const img = document.createElement('img');
        img.src = e.target.result;
        previewElement.appendChild(img);
        
        // Store image data
        if (imageNumber === 1) {
            image1 = e.target.result;
        } else {
            image2 = e.target.result;
        }
        
        // Hide any previous errors
        hideError();
    };
    
    reader.readAsDataURL(file);
}

function predictSimilarity() {
    if (!image1 || !image2) {
        showError('Please upload both images first');
        return;
    }

    // Show loading indicator
    loadingElement.style.display = 'block';
    resultElement.textContent = 'Processing...';
    resultDetailsElement.textContent = '';
    hideError();
    
    // Send images to backend API
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            image1: image1,
            image2: image2
        })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'An error occurred during prediction');
            });
        }
        return response.json();
    })
    .then(data => {
        resultElement.textContent = `${data.similarity.toFixed(2)}% Similar`;
        
        if (data.verified) {
            resultDetailsElement.textContent = 'High similarity detected. These appear to be the same person.';
            resultElement.style.color = '#27ae60';
        } else if (data.similarity > 60) {
            resultDetailsElement.textContent = 'Moderate similarity detected. These might be the same person.';
            resultElement.style.color = '#f39c12';
        } else {
            resultDetailsElement.textContent = 'Low similarity detected. These are likely different people.';
            resultElement.style.color = '#e74c3c';
        }
    })
    .catch(error => {
        showError(error.message);
        resultElement.textContent = 'Analysis failed';
        resultDetailsElement.textContent = '';
    })
    .finally(() => {
        loadingElement.style.display = 'none';
    });
}

function showError(message) {
    errorElement.textContent = message;
    errorElement.style.display = 'block';
}

function hideError() {
    errorElement.style.display = 'none';
}