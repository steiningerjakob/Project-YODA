import pandas as pd
from PIL import Image as PImage
from projectYoda.params import root_dir

def get_dataframes():
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


def load_images(root_dir, df):
    '''Load images from root director and path from datafram'''

    loaded_images = [PImage.open(root_dir + image) for image in df['path']]

    return loaded_images
