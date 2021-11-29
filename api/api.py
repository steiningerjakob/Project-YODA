from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from PIL import Image
#import segmentation_models as sm
from projectYoda.data import get_test_data
from projectYoda.gcp import get_model_from_gcp
from google.cloud import storage
from io import BytesIO

from tensorflow import keras

### deep learning imports
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, Activation
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import utils


IMAGE_URL = 'https://storage.cloud.google.com/wagon-data-745-project-yoda/images/image.jpg'
BUCKET_NAME = 'wagon-data-745-project-yoda'
DESTINATION_BLOB_NAME = 'images/image.jpg'
DESTINATION_FILE_NAME = 'image.jpg'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


def download_blob():
    """Downloads a blob from the bucket."""

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(BUCKET_NAME)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(DESTINATION_BLOB_NAME)
    blob.download_to_filename(DESTINATION_FILE_NAME)

    return blob


def preprocess_test_image(image):
    '''Transform test image into vector for model prediction'''

    # resize image to 224x224 pixels
    test_image = image.resize((224, 224))
    # represent image as numpy array
    pixels = np.asarray(test_image)
    # convert from integers to floats
    pixels = pixels.astype('float32')
    # normalize to the range 0-1
    pixels /= 255.0
    # reshape numpy array to pass into model
    test_image = np.reshape(
        pixels,(1, 224, 224,3))  # 1 image, (224, 224) size, 3 representing the RGB type.

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


@app.get('/predict')
def predict():

    # save image to local directory as blob
    download_blob()

    # open image from local directory as Pillow object
    image = Image.open('./image.jpg')

    # preprocess image for input into model (resize, reshape, rescale)
    test_image = preprocess_test_image(image)

    # get test df for minifigure name mapping
    df_test = get_test_data()

    # get model locally - from root directory
    model = get_model(source='local')

    # make prediction
    result = model.predict(test_image) # returns an array of probabilities

    # convert prediction class into minifigure name
    prediction_class = result.argmax() # returns the class with the highest probability
    prediction_character = df_test['minifigure_name'].iloc[prediction_class]

    # return minifigure name as JSON for frontend:
    return dict(prediction=prediction_character)


if __name__ == "__main__":
    prediction = predict()
    print(prediction)
