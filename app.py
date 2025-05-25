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
logging.basicConfig(level=logging.DEBUG, # Changed to DEBUG for more verbose output during debugging
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
    """Initializes and caches the DeepFace instance."""
    try:
        from deepface import DeepFace
        logger.info("DeepFace initialized successfully.")
        return DeepFace
    except ImportError:
        logger.error("DeepFace library not found. Please install it with 'pip install deepface'.")
        return None
    except Exception as e:
        logger.error(f"Error loading DeepFace: {str(e)} - {traceback.format_exc()}")
        return None

# Available models and detectors
AVAILABLE_MODELS = [
    {"name": "Facenet512", "accuracy": 98.4, "recommended": True},
    {"name": "Facenet", "accuracy": 97.4, "recommended": False},
    {"name": "Dlib", "accuracy": 96.8, "recommended": False}, # This refers to Dlib recognition model
    {"name": "VGG-Face", "accuracy": 96.7, "recommended": False},
    {"name": "ArcFace", "accuracy": 96.7, "recommended": False}
]

AVAILABLE_DETECTORS = [
    {"name": "retinaface", "description": "Most Accurate", "recommended": True},
    {"name": "mediapipe", "description": "", "recommended": False},
    {"name": "mtcnn", "description": "", "recommended": False},
    {"name": "dlib", "description": "", "recommended": False}, # This refers to Dlib face detector
    {"name": "ssd", "description": "", "recommended": False},
    {"name": "opencv", "description": "", "recommended": False} # Added opencv explicitly as it's often the default
]

DISTANCE_METRICS = [
    {"name": "cosine", "recommended": True},
    {"name": "euclidean_l2", "recommended": False},
    {"name": "euclidean", "recommended": False}
]

@app.route('/')
def index():
    return render_template('index.html')

def preprocess_image(img_data, target_filename):
    """Decodes base64 image data, resizes it if too large, and saves to a file."""
    try:
        if ',' in img_data:
            img_data = img_data.split(',')[1] # Remove data URL prefix
        
        # Decode base64 to numpy array
        img_array = np.frombuffer(base64.b64decode(img_data), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error(f"Failed to decode image from base64 data. Length: {len(img_data)}")
            return None
        
        # Resize if too large to prevent memory issues and speed up processing
        max_dimension = 1024 # Increased max dimension for better quality on larger screens
        height, width = img.shape[:2]
        if height > max_dimension or width > max_dimension:
            scale = max_dimension / max(height, width)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        cv2.imwrite(target_filename, img)
        logger.debug(f"Image preprocessed and saved to {target_filename}")
        return target_filename
    except Exception as e:
        logger.error(f"Error in preprocessing image: {str(e)} - {traceback.format_exc()}")
        return None

def analyze_face_attributes(img_path, primary_detector_backend):
    """
    Analyze face attributes like age, gender, race, emotion using a specified detector.
    Includes fallback detectors if the primary one fails.
    """
    try:
        DeepFace = get_deepface()
        if not DeepFace:
            logger.warning("DeepFace not available for attribute analysis.")
            return None
            
        # Define a list of detectors to try, starting with the primary one.
        # Adding more robust fallbacks if the primary is not already one of them.
        detectors_to_try = [primary_detector_backend]
        if primary_detector_backend != "retinaface":
            detectors_to_try.append("retinaface") # RetinaFace is generally very good
        if primary_detector_backend != "mtcnn":
            detectors_to_try.append("mtcnn")      # MTCNN is also quite robust
        if primary_detector_backend != "dlib": # dlib also good, but can be slow
            detectors_to_try.append("dlib")
        if primary_detector_backend != "opencv":
            detectors_to_try.append("opencv") # Always keep opencv as a final lightweight fallback

        analysis_result_dict = None
        for current_detector in list(dict.fromkeys(detectors_to_try)): # Remove duplicates while preserving order
            try:
                logger.debug(f"Attempting attribute analysis for {img_path} with detector: {current_detector}")
                analysis_list = DeepFace.analyze(
                    img_path=img_path,
                    actions=['age', 'gender', 'race', 'emotion'],
                    enforce_detection=False, # Set to False to allow analysis even if detection is weak
                    detector_backend=current_detector
                )
                
                # DeepFace.analyze returns a list of dictionaries.
                # If no face is detected, it returns an empty list [].
                # If enforce_detection=False, it might return [None] or similar if detection is very weak.
                if analysis_list and isinstance(analysis_list, list) and len(analysis_list) > 0:
                    # Filter out any None results if enforce_detection=False returns them
                    valid_analyses = [a for a in analysis_list if a is not None and a.get('region')]
                    if valid_analyses:
                        analysis_result_dict = valid_analyses[0] # Take the first detected face's attributes
                        logger.info(f"Attribute analysis successful for {img_path} with detector: {current_detector}")
                        break # Stop trying other detectors if successful
                    else:
                        logger.warning(f"No valid face region detected by {current_detector} for attributes in {img_path}.")
                else:
                    logger.warning(f"Analysis returned empty list for {img_path} with detector: {current_detector}.")
                
            except ValueError as ve: # Specific error for no face detected by DeepFace
                logger.warning(f"ValueError from {current_detector} for attributes in {img_path}: {str(ve)}")
            except Exception as e:
                logger.error(f"Error during DeepFace.analyze for attributes with {current_detector} on {img_path}: {str(e)}")

        if not analysis_result_dict: # All detectors failed to provide an analysis result
            logger.warning(f"All attribute detectors failed for {img_path}. Returning None for attributes.")
            return None
            
        # Extract and clean data for response
        # DeepFace provides dominant_gender, dominant_race, dominant_emotion directly
        # Ensure default values are returned if an attribute isn't found
        result = {
            'age': int(analysis_result_dict.get('age', 0)), # Age is a direct number
            'gender': str(analysis_result_dict.get('dominant_gender', 'Unknown')),
            'race': str(analysis_result_dict.get('dominant_race', 'Unknown')),
            'emotion': str(analysis_result_dict.get('dominant_emotion', 'Unknown')),
            
            # Confidence scores are in a nested dictionary and need to be multiplied by 100
            # Use default of 0 if dominant category or dictionary is missing
            'gender_confidence': float(analysis_result_dict.get('gender', {}).get(analysis_result_dict.get('dominant_gender', 'Unknown'), 0) * 100) if analysis_result_dict.get('dominant_gender') else 0,
            'race_confidence': float(analysis_result_dict.get('race', {}).get(analysis_result_dict.get('dominant_race', 'Unknown'), 0) * 100) if analysis_result_dict.get('dominant_race') else 0,
            'emotion_confidence': float(analysis_result_dict.get('emotion', {}).get(analysis_result_dict.get('dominant_emotion', 'Unknown'), 0) * 100) if analysis_result_dict.get('dominant_emotion') else 0
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Top-level error analyzing face attributes for {img_path}: {str(e)} - {traceback.format_exc()}")
        return None

def detect_image_type(img_path):
    """Detect if image is anime, AI-generated, or real photo using improved heuristics."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Could not read image for type detection: {img_path}")
            return "Unknown"
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Feature 1: Edge density/sharpness (Laplacian Variance)
        # Higher variance indicates sharper details, common in AI-gen or crisp photos/anime
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Feature 2: Color Saturation Mean (HSV S channel)
        # Anime/Cartoons often have higher average saturation and vibrant colors
        saturation_mean = np.mean(hsv[:,:,1])

        # Feature 3: Number of unique colors (simple proxy for color diversity/flatness)
        # Use a subsample for large images to avoid memory issues and speed up
        sample_size = (200, 200) # Reduce sample size for faster processing
        sample_img = cv2.resize(img, sample_size, interpolation=cv2.INTER_AREA)
        num_unique_colors = len(np.unique(sample_img.reshape(-1, sample_img.shape[2]), axis=0))

        # Feature 4: Flatness/Smoothness (e.g., variance of color channels)
        # Low variance can suggest flat shading (anime)
        color_channel_variance = np.mean([np.var(img[:,:,i]) for i in range(3)])

        logger.debug(f"Image Type Detection Metrics for {img_path}: laplacian_var={laplacian_var:.2f}, saturation_mean={saturation_mean:.2f}, num_unique_colors={num_unique_colors}, color_channel_variance={color_channel_variance:.2f}")

        # --- Heuristics based on common observations ---
        # Prioritize Anime/Cartoon first as it has distinct visual characteristics
        # Adjusted thresholds for better anime/cartoon detection
        if (saturation_mean > 100 and laplacian_var > 300 and color_channel_variance < 800) or \
           (saturation_mean > 150 and num_unique_colors < 5000): # Very vibrant and limited colors
            return "Anime/Cartoon"

        # Next, check for AI Generated (suspected) characteristics
        # AI images often have very high detail, unnaturally sharp edges, or specific 'texture'
        # These thresholds are still heuristic and can be tuned.
        if (laplacian_var > 4000) or \
           (num_unique_colors > (img.shape[0]*img.shape[1]/50) and laplacian_var > 2000): # Many colors and high sharpness
            return "AI Generated (suspected)"
        
        # If none of the above, it's likely a Real Photo
        return "Real Photo"
            
    except Exception as e:
        logger.error(f"Error detecting image type for {img_path}: {str(e)} - {traceback.format_exc()}")
        return "Unknown"

def verify_faces_advanced(img1_path, img2_path, model_name='Facenet', detector_backend='opencv', distance_metric='cosine'):
    """Performs advanced face verification with specified parameters."""
    try:
        DeepFace = get_deepface()
        if not DeepFace:
            raise Exception("DeepFace library not available. Cannot perform verification.")
        
        logger.info(f"Attempting DeepFace.verify with model='{model_name}', detector='{detector_backend}', metric='{distance_metric}'")
        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=False, # Set to False to allow processing even if a face isn't clearly detected (but might lead to higher distance)
            distance_metric=distance_metric
        )
        logger.info(f"DeepFace verification result: {result}")
        
        return result
        
    except ValueError as ve: # Specific error for no face detected by DeepFace.verify
        logger.warning(f"DeepFace failed to detect face(s) for verification with {detector_backend}: {str(ve)}. Attempting fallback.")
        raise ve # Re-raise to trigger fallback
    except Exception as e:
        logger.error(f"Advanced verification failed: {str(e)} - {traceback.format_exc()}")
        raise e

def calculate_histogram_similarity(img1_path, img2_path):
    """Fallback histogram-based similarity for when DeepFace fails."""
    try:
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            raise ValueError("Could not read one or both images for histogram comparison.")

        # Convert to RGB and resize for consistent histogram calculation
        size = (224, 224)
        img1_resized = cv2.resize(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), size)
        img2_resized = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), size)
        
        # Calculate 3D histograms (for R, G, B channels)
        # Bins: 8 for each channel (8*8*8 = 512 bins)
        # Range: 0-256 for each channel
        hist1 = cv2.calcHist([img1_resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2_resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        # Normalize histograms
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        # Compare histograms using correlation method
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        similarity_percentage = float(max(0, min(100, correlation * 100))) # Ensure between 0-100
        
        logger.info(f"Histogram similarity fallback used. Correlation: {correlation}, Similarity: {similarity_percentage}%")
        
        return {
            'similarity': float(similarity_percentage),
            'verified': bool(similarity_percentage > 60), # A simple threshold for histogram
            'distance': float(1 - correlation),
            'threshold': float(0.4), # Default for histogram fallback
            'model': 'Histogram',
            'detector_backend': 'N/A',
            'similarity_metric': 'correlation'
        }
        
    except Exception as e:
        logger.error(f"Histogram similarity failed: {str(e)} - {traceback.format_exc()}")
        return None

def save_analysis_result(data, result_id):
    """Save analysis result for sharing."""
    try:
        result_file = os.path.join(RESULTS_DIR, f"{result_id}.json")
        with open(result_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Analysis result saved to {result_file}")
        return result_id
    except Exception as e:
        logger.error(f"Error saving result: {str(e)} - {traceback.format_exc()}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    img1_filename, img2_filename = None, None # Initialize to None for cleanup in case of early error
    try:
        logger.info("Starting advanced prediction request.")
        
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        img1_data = data.get('image1')
        img2_data = data.get('image2')
        
        # Default values if not provided from frontend
        model_name = data.get('model', 'Facenet512') 
        detector_backend = data.get('detector', 'opencv')
        distance_metric = data.get('distance_metric', 'cosine')
        
        if not img1_data or not img2_data:
            return jsonify({'error': 'Please provide both images'}), 400
        
        # Validate parameters against the defined lists
        valid_model_names = [m['name'] for m in AVAILABLE_MODELS]
        if model_name not in valid_model_names:
            logger.warning(f"Invalid model_name received: {model_name}. Defaulting to 'Facenet512'.")
            model_name = 'Facenet512'

        valid_detector_names = [d['name'] for d in AVAILABLE_DETECTORS]
        if detector_backend not in valid_detector_names:
            logger.warning(f"Invalid detector_backend received: {detector_backend}. Defaulting to 'opencv'.")
            detector_backend = 'opencv'
        
        valid_distance_metrics = [m['name'] for m in DISTANCE_METRICS]
        if distance_metric not in valid_distance_metrics:
            logger.warning(f"Invalid distance_metric received: {distance_metric}. Defaulting to 'cosine'.")
            distance_metric = 'cosine'

        logger.info(f"Using config: Model={model_name}, Detector={detector_backend}, Metric={distance_metric}")
        
        # Create unique filenames for temporary storage
        img1_filename = os.path.join(TEMP_DIR, f"img1_{uuid.uuid4().hex}.jpg")
        img2_filename = os.path.join(TEMP_DIR, f"img2_{uuid.uuid4().hex}.jpg")
        
        # Preprocess images
        img1_path = preprocess_image(img1_data, img1_filename)
        img2_path = preprocess_image(img2_data, img2_filename)
        
        if not img1_path or not img2_path:
            cleanup_files(img1_filename, img2_filename)
            return jsonify({'error': 'Failed to process images. Ensure they are valid image files.'}), 400
        
        # Analyze face attributes for both images
        logger.info("Analyzing face attributes.")
        face1_attributes = analyze_face_attributes(img1_path, detector_backend)
        face2_attributes = analyze_face_attributes(img2_path, detector_backend)
        
        # Detect image types
        img1_type = detect_image_type(img1_path)
        img2_type = detect_image_type(img2_path)
        
        # Face similarity analysis
        logger.info("Starting face similarity analysis (DeepFace.verify).")
        
        result = {} # Initialize result dictionary for verification
        try:
            result = verify_faces_advanced(
                img1_path, img2_path, model_name, detector_backend, distance_metric
            )
            
            # DeepFace.verify returns 'distance', 'threshold', 'verified', 'model', 'detector_backend', 'similarity_metric'
            distance = float(result.get('distance', 0))
            threshold = float(result.get('threshold', 0.4))
            verified = bool(result.get('verified', False))
            
            # Calculate similarity percentage from distance
            if distance_metric == 'cosine':
                similarity_percentage = round(max(0, min(100, (1 - min(distance, 1)) * 100)), 2)
            else: # Euclidean or Euclidean_l2 distance
                normalized_distance = min(distance / 2.0, 1.0) # Assume max meaningful distance is around 2.0
                similarity_percentage = round(max(0, min(100, (1 - normalized_distance) * 100)), 2)
            
            method_used = "DeepFace"
            
        except Exception as face_error:
            logger.warning(f"DeepFace verification failed ({model_name}, {detector_backend}): {str(face_error)}. Falling back to Histogram Comparison.")
            result = calculate_histogram_similarity(img1_path, img2_path)
            
            if not result:
                cleanup_files(img1_filename, img2_filename)
                return jsonify({'error': 'Unable to analyze images using DeepFace or fallback method. Faces might be too unclear.'}), 500
            
            # Update values based on histogram result for final response
            similarity_percentage = result['similarity']
            distance = result['distance']
            threshold = result['threshold']
            verified = result['verified']
            method_used = "Histogram Comparison"
            # Overwrite model/detector/metric to reflect fallback
            model_name = result['model'] # 'Histogram'
            detector_backend = result['detector_backend'] # 'N/A'
            distance_metric = result['similarity_metric'] # 'correlation'

        # Generate result ID for sharing
        result_id = str(uuid.uuid4())
        
        # Prepare comprehensive response
        response = {
            'similarity': float(similarity_percentage),
            'verified': bool(verified),
            'distance': float(distance),
            'threshold': float(threshold),
            'method_used': str(method_used),
            'model_name': str(result.get('model', model_name)),
            'detector_backend': str(result.get('detector_backend', detector_backend)),
            'distance_metric': str(result.get('similarity_metric', distance_metric)),
            'face1_attributes': face1_attributes,
            'face2_attributes': face2_attributes,
            'img1_type': str(img1_type),
            'img2_type': str(img2_type),
            'analysis_timestamp': datetime.now().isoformat(),
            'result_id': result_id
        }
        
        # Save result for sharing
        save_analysis_result(response, result_id)
        
        # Clean up temporary files
        cleanup_files(img1_filename, img2_filename)
        
        logger.info(f"Prediction complete for result_id: {result_id}")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"An unexpected error occurred during prediction: {str(e)} - {traceback.format_exc()}")
        cleanup_files(img1_filename, img2_filename) # Ensure cleanup even on unexpected errors
        return jsonify({'error': 'An unexpected error occurred. Please check server logs for details.'}), 500

@app.route('/get_models')
def get_models():
    """Returns available models, detectors, and distance metrics to the frontend."""
    return jsonify({
        'models': AVAILABLE_MODELS,
        'detectors': AVAILABLE_DETECTORS,
        'distance_metrics': DISTANCE_METRICS
    })

@app.route('/share/<result_id>')
def share_result(result_id):
    """Renders a page for sharing a saved analysis result."""
    try:
        result_file = os.path.join(RESULTS_DIR, f"{result_id}.json")
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                data = json.load(f)
            logger.info(f"Serving share page for result_id: {result_id}")
            return render_template('share.html', result=data)
        else:
            logger.warning(f"Share result not found for ID: {result_id}")
            return render_template('share_not_found.html'), 404
    except Exception as e:
        logger.error(f"Error loading shared result {result_id}: {str(e)} - {traceback.format_exc()}")
        return render_template('share_error.html'), 500

@app.route('/test_deepface')
def test_deepface():
    """Tests if DeepFace is initialized and reports available configurations."""
    try:
        DeepFace = get_deepface()
        if DeepFace is None:
            return jsonify({'status': 'error', 'message': 'DeepFace library is not available or failed to initialize.'}), 500
        
        return jsonify({
            'status': 'success',
            'message': 'DeepFace library loaded and ready.',
            'available_models': AVAILABLE_MODELS,
            'available_detectors': AVAILABLE_DETECTORS,
            'available_distance_metrics': DISTANCE_METRICS
        })
    except Exception as e:
        logger.error(f"Error during DeepFace test: {str(e)} - {traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': f'An error occurred while testing DeepFace: {str(e)}'}), 500

def cleanup_files(*file_paths):
    """Removes temporary files."""
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug(f"Cleaned up temporary file: {file_path}")
            except OSError as e:
                logger.error(f"Error cleaning up file {file_path}: {str(e)}")

if __name__ == '__main__':
    # Initialize DeepFace once when the app starts
    logger.info("Attempting to initialize DeepFace on app startup...")
    _ = get_deepface() 
    
    app.run(debug=True) # Set debug=False for production