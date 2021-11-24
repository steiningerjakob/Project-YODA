from google.cloud import storage
import pandas as pd
from PIL import Image as PImage
from projectYoda.params import root_dir, BUCKET_NAME, BUCKET_DATA_PATH

def get_dataframes_locally():
    '''Get train and validation dataframes'''
    index = pd.read_csv('../raw_data/index.csv')
    metadata = pd.read_csv('../raw_data/metadata.csv')
    df = pd.merge(index, metadata, on='class_id')

    # put all but 2 images of each class in training set
    df_train = [
        df[df['class_id'] == i].iloc[:-2]
        for i in range(1,
                       len(df['class_id'].value_counts()) + 1)
    ]

    # put 2 images of each class in validation set
    df_valid = [
        df[df['class_id'] == i].iloc[-2:]
        for i in range(1,
                       len(df['class_id'].value_counts()) + 1)
    ]

    return df_train, df_valid


def get_dataframes_from_gcp(optimize=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    client = storage.Client()
    base_path = f"gs://{BUCKET_NAME}/{BUCKET_DATA_PATH}"
    index = pd.read_csv(f'{base_path}/index.csv')
    metadata = pd.read_csv(f'{base_path}/metadata.csv')
    df = pd.merge(index, metadata, on='class_id')

    # put all but 2 images of each class in training set
    df_train = [
        df[df['class_id'] == i].iloc[:-2]
        for i in range(1,
                       len(df['class_id'].value_counts()) + 1)
    ]

    # put 2 images of each class in validation set
    df_valid = [
        df[df['class_id'] == i].iloc[-2:]
        for i in range(1,
                       len(df['class_id'].value_counts()) + 1)
    ]

    return df_train, df_valid


def load_images(root_dir, df):
    '''Load images from root director and path from datafram'''

    loaded_images = [PImage.open(root_dir + image) for image in df['path']]

    return loaded_images

if __name__ == "__main__":
    df_train, df_valid = get_dataframes_from_gcp()
    print(type(df_valid))
