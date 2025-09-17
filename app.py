from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the trained model
model = load_model('Blood_Cell.keras')

# Define class names (must match your model training order)
class_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No file selected", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(128, 128))  # Match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 128, 128, 3)
    img_array /= 255.0

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)

    return render_template(
        'result.html',
        predicted_class=predicted_class,
        confidence=confidence,
        image_path=filepath
    )

if __name__ == '__main__':
    app.run(debug=True)
