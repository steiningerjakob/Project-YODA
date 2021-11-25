import os
from google.cloud import storage
from termcolor import colored
from projectYoda.params import BUCKET_NAME, MODEL_NAME, MODEL_VERSION, PATH_TO_LOCAL_MODEL
import joblib


LOCAL_MODEL_NAME = PATH_TO_LOCAL_MODEL

def store_model_on_gcp(rm=False):
    '''Upload model to Google Cloud Storage'''
    client = storage.Client().bucket(BUCKET_NAME)

    storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{LOCAL_MODEL_NAME}"
    blob = client.blob(storage_location)
    blob.upload_from_filename('model.joblib')
    print(colored(f"=> model.joblib uploaded to bucket {BUCKET_NAME} inside {storage_location}",
                  "green"))
    if rm:
        os.remove('model.joblib')


def get_model_from_gcp(rm=False):
    """Get the model from Google Cloud Storage"""
    storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{LOCAL_MODEL_NAME}"
    path = f"gs://{BUCKET_NAME}/{storage_location}"
    model = joblib.load(path)
    return model
