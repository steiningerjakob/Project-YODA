from google.cloud import storage
import pandas as pd
from PIL import Image as PImage
from projectYoda.params import BUCKET_NAME, BUCKET_DATA_PATH


def get_train_data(source='gcp', optimize=False, **kwargs):
    """method to get the training data from google cloud bucket or locally"""
    if source == 'gcp':
        base_path = f"gs://{BUCKET_NAME}/{BUCKET_DATA_PATH}"
    else:
        base_path = '../raw_data' # local path

    index = pd.read_csv(f'{base_path}/index.csv')
    metadata = pd.read_csv(f'{base_path}/metadata.csv')
    df = pd.merge(index, metadata, on='class_id')

    # put all but 2 images of each class in training set, rest in validation set
    df_train = pd.DataFrame([])
    df_valid = pd.DataFrame([])
    for i in range(1, len(df['class_id'].value_counts()) + 1):
        df_train = df_train.append(df[df['class_id'] == i].iloc[:-2])
        df_valid = df_valid.append(df[df['class_id'] == i].iloc[-2:])

    return df_train, df_valid


def get_test_data(**kwargs):
    """method to get the test data from google cloud bucket"""
    base_path = f"gs://{BUCKET_NAME}/{BUCKET_DATA_PATH}"
    test = pd.read_csv(f'{base_path}/test.csv')
    metadata = pd.read_csv(f'{base_path}/metadata.csv')
    df_test = pd.merge(test,
                       metadata[['class_id', 'minifigure_name']],
                       on='class_id')

    return df_test


def get_metadata(**kwargs):
    """method to get the test data from google cloud bucket"""
    base_path = f"gs://{BUCKET_NAME}/{BUCKET_DATA_PATH}"
    metadata = pd.read_csv(f'{base_path}/metadata.csv')

    return metadata


def load_images(root_dir, df):
    '''Load images from root director and path from datafram'''

    loaded_images = [PImage.open(root_dir + image) for image in df['path']]

    return loaded_images


if __name__ == "__main__":
    df_train, df_valid = get_train_data(source='gcp')
    print(df_valid.head())
