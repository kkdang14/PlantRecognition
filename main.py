import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from flask import Flask, request, render_template, redirect, url_for, flash
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
MODEL_PATH = 'plant_recognition_model_1.h5'
model = load_model(MODEL_PATH)

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess image for the model
def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224))  # Resize image to 224x224
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for predicting the plant type
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
    # if file and allowed_file(file.filename):
    #     filename = secure_filename(file.filename)
    #     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    #     file.save(file_path)

        # Preprocess the image
        image = preprocess_image(file_path)

        # Predict the class using the model
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)[0]  # Get the class index

        # (Assuming you have a class dictionary to map class index to plant names)
        class_dict = {0: 'Dương Xỉ', 1: 'Lan Ý', 2: 'Lưỡi Hổ', 3: 'Thủy Tùng', 4: 'Vạn Niên Thanh'}
        plant_name = class_dict.get(predicted_class, "Unknown Plant")

        return render_template('result.html', plant_name=plant_name)

    else:
        flash('Allowed image types are -> png, jpg, jpeg')
        return redirect(request.url)

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=False, host="0.0.0.0")