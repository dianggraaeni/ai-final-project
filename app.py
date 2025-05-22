from flask import Flask, render_template, request, jsonify, send_file
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
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Create directories
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_uploads')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Cache DeepFace model
@lru_cache(maxsize=1)
def get_deepface():
    try:
        from deepface import DeepFace
        logger.info("DeepFace initialized successfully")
        return DeepFace
    except Exception as e:
        logger.error(f"Error loading DeepFace: {str(e)}")
        return None

# Available models and detectors
AVAILABLE_MODELS = ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']
AVAILABLE_DETECTORS = ['opencv', 'retinaface', 'mtcnn', 'ssd', 'dlib', 'mediapipe']
DISTANCE_METRICS = ['cosine', 'euclidean', 'euclidean_l2']

@app.route('/')
def index():
    return render_template('index.html')

def preprocess_image(img_data, target_filename):
    try:
        if ',' in img_data:
            img_data = img_data.split(',')[1]
        
        img_array = np.frombuffer(base64.b64decode(img_data), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error("Failed to decode image")
            return None
        
        # Resize if too large
        max_dimension = 800
        height, width = img.shape[:2]
        if height > max_dimension or width > max_dimension:
            scale = max_dimension / max(height, width)
            img = cv2.resize(img, None, fx=scale, fy=scale)
        
        cv2.imwrite(target_filename, img)
        return target_filename
    except Exception as e:
        logger.error(f"Error in preprocessing image: {str(e)}")
        return None

def analyze_face_attributes(img_path):
    """Analyze face attributes like age, gender, race, emotion"""
    try:
        DeepFace = get_deepface()
        if not DeepFace:
            return None
            
        # Analyze face attributes
        analysis = DeepFace.analyze(
            img_path=img_path,
            actions=['age', 'gender', 'race', 'emotion'],
            enforce_detection=False
        )
        
        # Handle both single face and multiple faces
        if isinstance(analysis, list):
            analysis = analysis[0]  # Take first face
        
        # Extract and clean data
        result = {
            'age': int(analysis.get('age', 0)),
            'gender': str(analysis.get('dominant_gender', 'Unknown')),
            'race': str(analysis.get('dominant_race', 'Unknown')),
            'emotion': str(analysis.get('dominant_emotion', 'Unknown')),
            'gender_confidence': float(analysis.get('gender', {}).get(analysis.get('dominant_gender', 'Woman'), 0)),
            'race_confidence': float(analysis.get('race', {}).get(analysis.get('dominant_race', 'asian'), 0)),
            'emotion_confidence': float(analysis.get('emotion', {}).get(analysis.get('dominant_emotion', 'neutral'), 0))
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing face attributes: {str(e)}")
        return None

def detect_image_type(img_path):
    """Detect if image is anime, AI-generated, or real photo"""
    try:
        img = cv2.imread(img_path)
        
        # Simple heuristics for image type detection
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Calculate various metrics
        saturation_mean = np.mean(hsv[:,:,1])
        brightness_variance = np.var(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        
        # Edge detection for sharpness analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Simple classification based on empirical thresholds
        if saturation_mean > 150 and laplacian_var > 1000:
            return "Anime/Cartoon"
        elif laplacian_var > 2000 and brightness_variance > 3000:
            return "AI Generated (suspected)"
        else:
            return "Real Photo"
            
    except Exception as e:
        logger.error(f"Error detecting image type: {str(e)}")
        return "Unknown"

def verify_faces_advanced(img1_path, img2_path, model_name='Facenet', detector_backend='opencv', distance_metric='cosine'):
    """Advanced face verification with custom parameters"""
    try:
        DeepFace = get_deepface()
        if not DeepFace:
            raise Exception("DeepFace not available")
        
        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=False,
            distance_metric=distance_metric
        )
        
        return result, model_name
        
    except Exception as e:
        logger.error(f"Advanced verification failed: {str(e)}")
        raise e

def calculate_histogram_similarity(img1_path, img2_path):
    """Fallback histogram-based similarity"""
    try:
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        size = (224, 224)
        img1_resized = cv2.resize(img1_rgb, size)
        img2_resized = cv2.resize(img2_rgb, size)
        
        hist1 = cv2.calcHist([img1_resized], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2_resized], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
        
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        similarity_percentage = float(max(0, correlation * 100))
        
        return {
            'similarity': float(similarity_percentage),
            'verified': bool(similarity_percentage > 50),
            'distance': float(1 - correlation),
            'threshold': float(0.5),
            'method': 'histogram'
        }
        
    except Exception as e:
        logger.error(f"Histogram similarity failed: {str(e)}")
        return None

def save_analysis_result(data, result_id):
    """Save analysis result for sharing"""
    try:
        result_file = os.path.join(RESULTS_DIR, f"{result_id}.json")
        with open(result_file, 'w') as f:
            json.dump(data, f, indent=2)
        return result_id
    except Exception as e:
        logger.error(f"Error saving result: {str(e)}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Starting advanced prediction request")
        
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        img1_data = data.get('image1')
        img2_data = data.get('image2')
        model_name = data.get('model', 'Facenet')
        detector_backend = data.get('detector', 'opencv')
        distance_metric = data.get('distance_metric', 'cosine')
        
        if not img1_data or not img2_data:
            return jsonify({'error': 'Please provide both images'}), 400
        
        # Validate parameters
        if model_name not in AVAILABLE_MODELS:
            model_name = 'Facenet'
        if detector_backend not in AVAILABLE_DETECTORS:
            detector_backend = 'opencv'
        if distance_metric not in DISTANCE_METRICS:
            distance_metric = 'cosine'
        
        # Create unique filenames
        img1_filename = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.jpg")
        img2_filename = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.jpg")
        
        # Preprocess images
        img1_path = preprocess_image(img1_data, img1_filename)
        img2_path = preprocess_image(img2_data, img2_filename)
        
        if not img1_path or not img2_path:
            cleanup_files(img1_filename, img2_filename)
            return jsonify({'error': 'Failed to process images'}), 400
        
        # Analyze face attributes for both images
        logger.info("Analyzing face attributes")
        face1_attributes = analyze_face_attributes(img1_path)
        face2_attributes = analyze_face_attributes(img2_path)
        
        # Detect image types
        img1_type = detect_image_type(img1_path)
        img2_type = detect_image_type(img2_path)
        
        # Face similarity analysis
        logger.info("Starting face similarity analysis")
        try:
            result, model_used = verify_faces_advanced(
                img1_path, img2_path, model_name, detector_backend, distance_metric
            )
            
            distance = float(result.get('distance', 0))
            threshold = float(result.get('threshold', 0.4))
            verified = bool(result.get('verified', False))
            
            # Calculate similarity percentage
            if distance_metric == 'cosine':
                similarity_percentage = round(max(0, min(100, (1 - min(distance, 1)) * 100)), 2)
            else:
                similarity_percentage = round(max(0, min(100, (1 - min(distance/4, 1)) * 100)), 2)
            
            method_used = f"DeepFace ({model_used})"
            
        except Exception as face_error:
            logger.warning(f"DeepFace failed, using histogram: {str(face_error)}")
            result = calculate_histogram_similarity(img1_path, img2_path)
            
            if not result:
                cleanup_files(img1_filename, img2_filename)
                return jsonify({'error': 'Unable to analyze images'}), 500
            
            similarity_percentage = result['similarity']
            distance = result['distance']
            threshold = result['threshold']
            verified = result['verified']
            method_used = "Histogram Comparison"
        
        # Generate result ID for sharing
        result_id = str(uuid.uuid4())
        
        # Prepare comprehensive response
        response = {
            'similarity': float(similarity_percentage),
            'verified': bool(verified),
            'distance': float(distance),
            'threshold': float(threshold),
            'method_used': str(method_used),
            'model_name': str(model_name),
            'detector_backend': str(detector_backend),
            'distance_metric': str(distance_metric),
            'face1_attributes': face1_attributes,
            'face2_attributes': face2_attributes,
            'img1_type': str(img1_type),
            'img2_type': str(img2_type),
            'analysis_timestamp': datetime.now().isoformat(),
            'result_id': result_id
        }
        
        # Save result for sharing
        save_analysis_result(response, result_id)
        
        # Clean up files
        cleanup_files(img1_filename, img2_filename)
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/get_models')
def get_models():
    """Get available models and detectors"""
    return jsonify({
        'models': AVAILABLE_MODELS,
        'detectors': AVAILABLE_DETECTORS,
        'distance_metrics': DISTANCE_METRICS
    })

@app.route('/share/<result_id>')
def share_result(result_id):
    """Share analysis result"""
    try:
        result_file = os.path.join(RESULTS_DIR, f"{result_id}.json")
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                data = json.load(f)
            return render_template('share.html', result=data)
        else:
            return render_template('share_not_found.html'), 404
    except Exception as e:
        logger.error(f"Error loading shared result: {str(e)}")
        return render_template('share_error.html'), 500

@app.route('/test_deepface')
def test_deepface():
    try:
        DeepFace = get_deepface()
        if DeepFace is None:
            return jsonify({'status': 'error', 'message': 'DeepFace not available'}), 500
        
        return jsonify({
            'status': 'success',
            'message': 'All systems operational',
            'available_models': AVAILABLE_MODELS,
            'available_detectors': AVAILABLE_DETECTORS
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def cleanup_files(file1, file2):
    """Remove temporary files"""
    try:
        for file_path in [file1, file2]:
            if os.path.exists(file_path):
                os.remove(file_path)
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")

if __name__ == '__main__':
    deepface = get_deepface()
    if deepface:
        logger.info("Advanced Face Analysis System Ready")
    else:
        logger.error("DeepFace initialization failed")
    
    app.run(debug=True)