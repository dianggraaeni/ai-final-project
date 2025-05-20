from flask import Flask, render_template, request, jsonify
import os
import base64
import numpy as np
import cv2
from deepface import DeepFace
import tempfile
import uuid
import re

app = Flask(__name__)

# Create temporary directory for image processing
TEMP_DIR = os.path.join(tempfile.gettempdir(), 'face_similarity')
os.makedirs(TEMP_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get base64 encoded images from request
        img1_data = request.json.get('image1')
        img2_data = request.json.get('image2')
        
        if not img1_data or not img2_data:
            return jsonify({'error': 'Please provide both images'}), 400
        
        # Clean the base64 data
        img1_data = re.sub('^data:image/.+;base64,', '', img1_data)
        img2_data = re.sub('^data:image/.+;base64,', '', img2_data)
        
        # Create unique filenames
        img1_filename = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.jpg")
        img2_filename = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.jpg")
        
        # Save base64 images to files
        with open(img1_filename, "wb") as f:
            f.write(base64.b64decode(img1_data))
        
        with open(img2_filename, "wb") as f:
            f.write(base64.b64decode(img2_data))
        
        # Verify both images contain detectable faces
        try:
            # Try to detect faces in both images
            face1 = DeepFace.detectFace(img_path=img1_filename)
            face2 = DeepFace.detectFace(img_path=img2_filename)
        except Exception as e:
            # Clean up files
            cleanup_files(img1_filename, img2_filename)
            return jsonify({'error': 'Could not detect faces in one or both images. Please use clear face images.'}), 400
        
        # Analyze similarity between faces
        try:
            result = DeepFace.verify(img1_path=img1_filename, 
                                    img2_path=img2_filename,
                                    model_name='VGG-Face',
                                    distance_metric='cosine')
            
            # Calculate percentage similarity (convert cosine distance to similarity percentage)
            # Cosine distance ranges from 0 (identical) to 1 (completely different)
            # Convert to similarity percentage where 0 distance = 100% similar
            distance = result['distance']
            similarity_percentage = round((1 - min(distance, 1)) * 100, 2)
            
            response = {
                'similarity': similarity_percentage,
                'verified': result['verified'],
                'distance': distance,
                'threshold': result['threshold']
            }
            
            # Clean up files
            cleanup_files(img1_filename, img2_filename)
            
            return jsonify(response), 200
            
        except Exception as e:
            # Clean up files
            cleanup_files(img1_filename, img2_filename)
            return jsonify({'error': f'Error during face verification: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

def cleanup_files(file1, file2):
    """Remove temporary files"""
    try:
        if os.path.exists(file1):
            os.remove(file1)
        if os.path.exists(file2):
            os.remove(file2)
    except Exception:
        pass  # Ignore cleanup errors

if __name__ == '__main__':
    app.run(debug=True)