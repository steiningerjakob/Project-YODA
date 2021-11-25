from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from projectYoda.data import get_test_data
from projectYoda.gcp import get_model_from_gcp

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/')
def index():
    return {'Yoda says:', 'You must unlearn what you have learned!'}


def preprocess_test_image(image):
    '''Transform test image into vector for model prediction'''

    test_image = image.resize(
        (256, 256))  # reshaping the image size to match training size
    test_image = np.reshape(test_image, (1, 256, 256, 3))
    # 1 image, (256, 256) size, 3 representing the RGB type.

    return test_image

def get_model(source='local'):
    if source == 'local':
        model = joblib.load('model.joblib')
    else:
        model = get_model_from_gcp(model='classification/baseline')

    return model


def predict(image):

    test_image = preprocess_test_image(image)
    df_test = get_test_data()
    model = get_model(source='gcp')

    result = model.predict(test_image) # returns an array of probabilities

    prediction_class = result.argmax() # returns the class with the highest probability
    prediction_character = df_test['minifigure_name'].iloc[prediction_class]

    # API Output:
    return dict(prediction=prediction_character)
