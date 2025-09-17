import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load model
model = tf.keras.models.load_model('blood_cell_model.h5')

# Class names (from training set)
class_names = os.listdir('dataset/prepared')

# Load and preprocess image
img_path = 'path/to/test_image.jpg'  # Replace with your image
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)
predicted_class = class_names[np.argmax(pred)]

print(f"Predicted class: {predicted_class}")
