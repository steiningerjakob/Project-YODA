import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"

import cv2
import base64
import pandas as pd
from io import BytesIO
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from PIL import Image
import segmentation_models as sm
from projectYoda.data import get_test_data
from projectYoda.gcp import get_model_from_gcp
from tensorflow import keras

### deep learning imports
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, Activation
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import utils

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


def preprocess_test_image(image):
    '''Transform test image into vector for model prediction'''

    test_image = image.resize((256, 256))
    pixels = np.asarray(test_image)
    # convert from integers to floats
    pixels = pixels.astype('float32')
    # normalize to the range 0-1
    pixels /= 255.0
    # confirm the normalization

    test_image = np.reshape(
        pixels,
        (1, 256, 256,
         3))  # 1 image, (256, 256) size, 3 representing the RGB type.
    print('test_image: ', test_image)

    return test_image


def get_model(source='local'):
    if source == 'local':
        model = joblib.load('model.joblib')
    else:
        model = get_model_from_gcp(model='classification/baseline')

    return model

@app.get('/')
def index():
    return {'Yoda says:', 'You must unlearn what you have learned!'}


@app.post('/predict')
def predict():

    with open('raw_data/test/001.jpg', "rb") as test_image:
        test_image_encoded = base64.b64encode(
            test_image.read())  # encoding the image to base64

    bytes = base64.b64decode(test_image_encoded)

    im_file = BytesIO(bytes)

    image = Image.open(im_file)

    test_image = preprocess_test_image(image)
    print(test_image)

    model = get_model(source='local')
    print(model)


    result = model.predict(test_image) # returns an array of probabilities
    print(result)

    prediction_class = result.argmax() # returns the class with the highest probability
    print(prediction_class)

    test = pd.read_csv("raw_data/test.csv")
    metadata = pd.read_csv("raw_data/metadata.csv")
    df_test = pd.merge(test,
                       metadata[['class_id', 'minifigure_name']],
                       on='class_id')

    prediction_character = df_test['minifigure_name'].iloc[prediction_class]

    # API Output:
    return dict(prediction=prediction_character)


if __name__ == "__main__":
    prediction = predict()
    print(prediction)
