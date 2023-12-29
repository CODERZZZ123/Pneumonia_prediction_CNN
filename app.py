import io
import numpy as np
from flask import Flask, render_template, request, send_file, send_from_directory
from tensorflow.keras.preprocessing import image
from PIL import Image
import tensorflow as tf
import io
from PIL import Image
from flask import Flask, render_template, request
from io import BytesIO
import base64

from time import time

app = Flask(__name__)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    timestamp = int(time())  # or use a random value
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename + '?' + str(timestamp))

# # Load the trained model
model = tf.keras.models.load_model("path/to/save/model_resnet_pneumonia")

# # Define the class names
class_names = ["normal", "bacteria_pneumonia", "virus_pneumonia"]  # Replace with your actual class names


def image_to_base64(img):
    img_pil = tf.keras.preprocessing.image.array_to_img(img[0])
    buffered = io.BytesIO()
    img_pil.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', prediction='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction='No selected file')

    # img_stream = file.stream
    # img = image.load_img(BytesIO(img_stream.read()), target_size=(64, 64))
    uploaded_file = request.files['file']

    # Save the uploaded file temporarily
    temp_path = 'temp_image.jpeg'
    uploaded_file.save(temp_path)

    class_names = ["normal", "bacteria_pneumonia", "virus_pneumonia"]  # Replace with your actual class names

    # Load and preprocess an example image for prediction
    # img_path = "data/chest_xray/train/pneumonia_virus/person1642_virus_2842.jpeg"
    img = image.load_img(temp_path, target_size=(64, 64))  # Adjust the target_size based on your model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values to the range [0, 1]


    actual_img = image.load_img(temp_path, target_size=(1000, 1000)) 
    actual_array = image.img_to_array(actual_img)
    actual_array = np.expand_dims(actual_array, axis=0)
    actual_array /= 255.0  # Normalize pixel values to the range [0, 1]
    
    # Make predictions
    predictions = model.predict(img_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions[0])

    # Get the predicted class label
    predicted_class_label = class_names[predicted_class_index]

    img_base64 = image_to_base64(actual_array)

    return render_template('index.html', prediction=predicted_class_label, image=img_base64)

if __name__ == '__main__':
    app.run(debug=True)
