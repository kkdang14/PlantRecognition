import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
from flask import Flask, request, render_template, redirect, url_for, flash, Response
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
from keras.api.models import load_model
import tensorflow as tf

# Define the Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'supersecretkey'

# Load the pre-trained model
MODEL_PATH = 'model/plant_recognition_model_1.h5'
model = load_model(MODEL_PATH)

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Webcam video capture
camera = cv2.VideoCapture(0)

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess image for the model
def preprocess_image(image):
    img = Image.fromarray(image).resize((224, 224))  # Resize image to 224x224
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to generate frames from webcam feed
def generate_frames():
    while True:
        success, frame = camera.read()  # Capture frame from the webcam
        if not success:
            break
        else:
            # Preprocess the frame for prediction
            preprocessed_frame = preprocess_image(frame)

            # Predict the class using the model
            predictions = model.predict(preprocessed_frame)
            predicted_class = np.argmax(predictions, axis=1)[0]  # Get the class index

            # Class dictionary (mapping the prediction index to plant name)
            class_dict = {0: 'duong_xi', 1: 'lan_y', 2: 'luoi_ho', 3: 'thuy_tung', 4: 'van_nien_thanh'}
            plant_name = class_dict.get(predicted_class, "Unknown Plant")

            # Display the prediction on the frame
            cv2.putText(frame, f'Plant: {plant_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Encode the frame as a JPEG image and yield it as byte data for real-time display
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for the home page
@app.route('/')
def home():
    return render_template('index2.html')

# Route for predicting the plant type from uploaded images
@app.route('/predict', methods=['POST'])
def predict():
    # Check if a valid image file is uploaded
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess the image
        image = preprocess_image(file_path)

        # Predict the class using the model
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)[0]  # Get the class index

        # Class dictionary
        class_dict = {0: 'duong_xi', 1: 'lan_y', 2: 'luoi_ho', 3: 'nha_dam', 4: 'thuy_tung', 5: 'van_nien_thanh'}  # Example
        plant_name = class_dict.get(predicted_class, "Unknown Plant")

        return render_template('result.html', plant_name=plant_name)

# Route for real-time plant recognition using webcam
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

