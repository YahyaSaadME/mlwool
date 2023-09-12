from tensorflow.keras.models import load_model
model = load_model('wool.h5')
from flask import Flask , request, jsonify
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
@app.route('/', methods=['GET'])
def welcome():
    return "Welcome"
    


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No file part"
    
    file = request.files['image']

    if file.filename == '':
        return "No selected file"

    # Specify the directory where you want to save the uploaded image
    upload_folder = 'uploads'
    file.save(os.path.join(upload_folder, file.filename))
    print(file.filename)
    input = Image.open(f"./uploads/{file.filename}")
    input = input.resize((256,256))
    pred = model.predict(np.expand_dims(input,axis=0))
    print(pred)
    final = pred.tolist()
    return jsonify(max(final[0]))

if __name__ == "__main__":
    app.run()