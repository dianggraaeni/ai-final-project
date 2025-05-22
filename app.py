from flask import Flask, render_template, request, jsonify
import os
import base64
import numpy as np
import cv2
import tempfile
import uuid
import re
from functools import lru_cache
import logging
import traceback
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Create local directory for image processing
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_uploads')
os.makedirs(TEMP_DIR, exist_ok=True)

# Cache DeepFace model to avoid reloading
@lru_cache(maxsize=1)
def get_deepface():
    try:
        from deepface import DeepFace
        logger.info("DeepFace initialized successfully")
        return DeepFace
    except Exception as e:
        logger.error(f"Error loading DeepFace: {str(e)}")
        return None

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Helper function to preprocess images
def preprocess_image(img_data, target_filename):
    try:
        # Clean the base64 data
        if ',' in img_data:
            img_data = img_data.split(',')[1]
        
        # Decode base64 to numpy array
        img_array = np.frombuffer(base64.b64decode(img_data), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error("Failed to decode image")
            return None
        
        # Resize to reasonable dimensions if too large
        max_dimension = 600  # Reduced from 800 for better performance
        height, width = img.shape[:2]
        if height > max_dimension or width > max_dimension:
            scale = max_dimension / max(height, width)
            img = cv2.resize(img, None, fx=scale, fy=scale)
            logger.debug(f"Image resized to {img.shape[1]}x{img.shape[0]}")
        
        # Save processed image
        cv2.imwrite(target_filename, img)
        logger.debug(f"Preprocessed image saved to {target_filename}")
        
        return target_filename
    except Exception as e:
        logger.error(f"Error in preprocessing image: {str(e)}")
        return None

# Multiple attempt face verification with different configurations
def verify_faces_multiple_attempts(img1_path, img2_path):
    """
    Try multiple configurations to verify faces
    """
    DeepFace = get_deepface()
    if not DeepFace:
        raise Exception("DeepFace not available")
    
    # Configuration attempts in order of preference
    configs = [
        # Most permissive - should work with most images
        {
            'model_name': 'Facenet',
            'detector_backend': 'opencv',
            'enforce_detection': False,
            'distance_metric': 'cosine'
        },
        # Alternative with different model
        {
            'model_name': 'VGG-Face',
            'detector_backend': 'opencv',
            'enforce_detection': False,
            'distance_metric': 'cosine'
        },
        # Try with RetinaFace detector
        {
            'model_name': 'Facenet',
            'detector_backend': 'retinaface',
            'enforce_detection': False,
            'distance_metric': 'cosine'
        },
        # Most basic - should work even with poor quality images
        {
            'model_name': 'Facenet',
            'detector_backend': 'opencv',
            'enforce_detection': False,
            'distance_metric': 'euclidean'
        }
    ]
    
    last_error = None
    
    for i, config in enumerate(configs):
        try:
            logger.info(f"Attempting verification with config {i+1}: {config}")
            
            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                **config
            )
            
            logger.info(f"Verification successful with config {i+1}")
            return result, config['model_name']
            
        except Exception as e:
            logger.warning(f"Config {i+1} failed: {str(e)}")
            last_error = e
            continue
    
    # If all configurations failed, raise the last error
    raise last_error

# Alternative simple similarity calculation
def calculate_simple_similarity(img1_path, img2_path):
    """
    Simple image similarity based on histogram comparison
    This is a fallback when face detection fails
    """
    try:
        # Read images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        # Convert to RGB
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Resize to same dimensions
        size = (224, 224)
        img1_resized = cv2.resize(img1_rgb, size)
        img2_resized = cv2.resize(img2_rgb, size)
        
        # Calculate histogram correlation
        hist1 = cv2.calcHist([img1_resized], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2_resized], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
        
        # Calculate correlation
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Convert to percentage and ensure all values are JSON serializable
        similarity_percentage = float(max(0, correlation * 100))
        
        return {
            'similarity': float(similarity_percentage),
            'verified': bool(similarity_percentage > 50),
            'distance': float(1 - correlation),
            'threshold': float(0.5),
            'method': 'histogram'
        }
        
    except Exception as e:
        logger.error(f"Simple similarity calculation failed: {str(e)}")
        return None

# Main prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Starting prediction request")
        
        # Get base64 encoded images from request
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        img1_data = data.get('image1')
        img2_data = data.get('image2')
        
        if not img1_data or not img2_data:
            return jsonify({'error': 'Please provide both images'}), 400
        
        # Create unique filenames
        img1_filename = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.jpg")
        img2_filename = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.jpg")
        
        # Preprocess and save images
        img1_path = preprocess_image(img1_data, img1_filename)
        img2_path = preprocess_image(img2_data, img2_filename)
        
        if not img1_path or not img2_path:
            cleanup_files(img1_filename, img2_filename)
            return jsonify({'error': 'Failed to process images'}), 400
        
        logger.info("Images processed, starting face verification")
        
        # Try face verification with multiple attempts
        try:
            result, model_used = verify_faces_multiple_attempts(img1_path, img2_path)
            
            logger.info(f"Face verification successful using {model_used}")
            
            # Extract results and convert numpy types to Python native types
            distance = float(result.get('distance', 0))
            threshold = float(result.get('threshold', 0.4))
            verified = bool(result.get('verified', False))
            
            # Calculate similarity percentage
            if 'cosine' in str(result):
                similarity_percentage = round(max(0, min(100, (1 - min(distance, 1)) * 100)), 2)
            else:
                # For euclidean distance, use different calculation
                similarity_percentage = round(max(0, min(100, (1 - min(distance/4, 1)) * 100)), 2)
            
            response = {
                'similarity': float(similarity_percentage),
                'verified': bool(verified),
                'distance': float(distance),
                'threshold': float(threshold),
                'model_used': str(model_used),
                'method': 'deepface'
            }
            
        except Exception as face_error:
            logger.warning(f"All DeepFace attempts failed: {str(face_error)}")
            
            # Fallback to simple similarity
            logger.info("Falling back to histogram-based similarity")
            result = calculate_simple_similarity(img1_path, img2_path)
            
            if result:
                response = result
                logger.info("Histogram similarity calculation successful")
            else:
                cleanup_files(img1_filename, img2_filename)
                return jsonify({
                    'error': 'Unable to analyze images. Please try with different images or ensure faces are clearly visible.'
                }), 500
        
        # Clean up files
        cleanup_files(img1_filename, img2_filename)
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Unexpected error in predict route: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500

def cleanup_files(file1, file2):
    """Remove temporary files"""
    try:
        if os.path.exists(file1):
            os.remove(file1)
        if os.path.exists(file2):
            os.remove(file2)
    except Exception as e:
        logger.error(f"Error cleaning up files: {str(e)}")

# Test route
@app.route('/test_deepface', methods=['GET'])
def test_deepface():
    try:
        DeepFace = get_deepface()
        if DeepFace is None:
            return jsonify({'status': 'error', 'message': 'DeepFace failed to load'}), 500
        
        return jsonify({
            'status': 'success', 
            'message': 'DeepFace is loaded and ready'
        })
    except Exception as e:
        logger.error(f"DeepFace test failed: {str(e)}")
        return jsonify({
            'status': 'error', 
            'message': f'DeepFace test failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Test DeepFace loading at startup
    deepface = get_deepface()
    if deepface:
        logger.info("DeepFace loaded successfully")
    else:
        logger.error("Failed to load DeepFace")
    
    app.run(debug=True)