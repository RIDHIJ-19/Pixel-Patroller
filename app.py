import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
import pickle
from PIL import Image
import io
import easyocr
import requests
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import cv2
import os
import shutil
from flask import Flask, render_template, request, jsonify, url_for
from PIL import Image
import numpy as np
import requests
import io
import os
import pickle
import easyocr

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models and resources
model_cb = pickle.load(open('decision_tree_model.pkl', 'rb'))
vectorizer = pickle.load(open("tfidfvectorizer.pkl", "rb"))
reader = easyocr.Reader(['en', 'hi'])
with open("stopwords.txt", "r") as file:
    stopwords = file.read().splitlines()

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/cb')
def cb_home():
    return render_template('cb_index.html', prediction=None, image_prediction=None, image_path=None, extracted_text=None)

@app.route('/cb/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        user_input = request.form['text']
        transformed_input = vectorizer.transform([user_input])
        prediction = model_cb.predict(transformed_input)[0]

    return render_template('cb_index.html', prediction=prediction, image_prediction=None, image_path=None, extracted_text=user_input)

@app.route('/cb/predict_text_api', methods=['POST'])
def predict_api():
    data = request.get_json()
    user_input = data.get('text', '')

    if user_input:
        transformed_input = vectorizer.transform([user_input])
        prediction = model_cb.predict(transformed_input)[0]
    else:
        prediction = "No input provided"

    return jsonify("1" if prediction == 1 else "0")

@app.route('/cb/predict_image', methods=['GET', 'POST'])
def predict_image():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Save the image to static folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
            file.save(filepath)

            # Extract text from the image
            image = Image.open(filepath)
            image = np.array(image)
            extracted_text = reader.readtext(image, detail=0)
            extracted_text = ' '.join(extracted_text)

            # Predict using extracted text
            response = requests.post('http://127.0.0.1:5000/cb/predict_text_api', json={'text': extracted_text})

            return render_template(
                'cb_index.html',
                image_prediction=response.json(),
                prediction=None,
                image_path=url_for('static', filename='uploaded_image.jpg'),
                extracted_text=extracted_text
            )
    return render_template('cb_index.html', prediction=None, image_prediction=None, image_path=None, extracted_text=None)


# ------------------------------------ DEEPFAKE MODEL ------------------------------------

image_dimensions = {'height':256, 'width':256, 'channels':3}

class Classifier:
    def __init__(self):
        self.model = 0


    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        return self.model.train_on_batch(x, y)

    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)

    def load(self, path):
        self.model.load_weights(path)

class Meso4(Classifier):
    def __init__(self, learning_rate = 0.001):
        self.model = self.init_model()
        optimizer = Adam(learning_rate = learning_rate)
        self.model.compile(optimizer = optimizer,
                           loss = 'mean_squared_error',
                           metrics = ['accuracy'])

    def init_model(self):
        x = Input(shape = (image_dimensions['height'],
                           image_dimensions['width'],
                           image_dimensions['channels']))

        x1 = Conv2D(8, (3, 3), padding='same', activation = 'relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        x2 = Conv2D(8, (5, 5), padding='same', activation = 'relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)

        return Model(inputs = x, outputs = y)
        
def clear_uploads():
    uploads_folder = "static/uploads"
    if not os.path.exists(uploads_folder):
        os.makedirs(uploads_folder)
    for file in os.listdir(uploads_folder):
        file_path = os.path.join(uploads_folder, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    try:
        shutil.rmtree(uploads_folder)
        os.makedirs(uploads_folder)
    except Exception as e:
        print(f"Error clearing uploads folder: {e}")
        
def detect_faces(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if not os.path.exists(image_path):
        return [], None
    image = cv2.imread(image_path)
    if image is None:
        return [], None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces, image
    
def preprocess_image(face):
    try:
        face = cv2.resize(face, (256, 256))  
        face = face / 255.0  
        face = np.expand_dims(face, axis=0)  
        return face
    except Exception as e:
        print("Error preprocessing image:", e)
        return None

model_df = Meso4()
model_df.load('model/model.h5')

@app.route('/df')
def dfhome():
    return render_template('df_index.html')

@app.route('/df/predict_df',methods=['GET','POST'])
def dfpredict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    file_path = os.path.join("static/uploads", file.filename)
    os.makedirs("static/uploads", exist_ok=True)  
    file.save(file_path)

    faces, image = detect_faces(file_path)
    if image is None:
        return jsonify({'error': 'Invalid image file. Please upload a valid image.'})

    if len(faces) == 0:
        return jsonify({'error': 'No face detected'})

    predictions = []
    for idx, (x, y, w, h) in enumerate(faces):
        x  = max(x - 50, 0)  
        y  = max(y - 50, 0)  
        w  = w + 2 * 50
        h = h + 2 * 50
        face = image[y:y+h, x:x+w]
        face_path = f"static/faces/face_{idx}.jpg"
        cv2.imwrite(face_path, face) 
        face = preprocess_image(face)
        prediction = model_df.predict(face)[0][0]
        result = "Real" if round(prediction) == 1 else "Deepfake"
        predictions.append({'face_location': (int(x), int(y), int(w), int(h)), 'prediction': result, 'face_url': face_path})

    image_url = f"/static/uploads/{file.filename}"
    return jsonify({'filename': file.filename, 'predictions': predictions, 'image_url': image_url})

if __name__ == '__main__':
    app.run(debug=True)
