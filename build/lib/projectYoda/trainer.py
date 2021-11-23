# TODO: transform into a class --> Trainer
# TODO: use pipelines for iteration and production

import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, Activation
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from projectYoda.data import get_dataframes
from projectYoda.params import root_dir


# Source for image data preprocessing
# https://vijayabhaskar96.medium.com/tutorial-on-keras-imagedatagenerator-with-flow-from-dataframe-8bd5776e45c1

def preprocess_data(df_train, df_valid, root_dir):
    '''Loads and preprocesses image data'''

    datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = datagen.flow_from_dataframe(dataframe=df_train,
                                                directory=root_dir,
                                                x_col="path",
                                                y_col="minifigure_name",
                                                class_mode="categorical",
                                                target_size=(256, 256),
                                                batch_size=16)
    valid_generator = datagen.flow_from_dataframe(dataframe=df_valid,
                                                directory=root_dir,
                                                x_col="path",
                                                y_col="minifigure_name",
                                                class_mode="categorical",
                                                target_size=(256, 256),
                                                batch_size=16,
                                                shuffle=False)
    return train_generator, valid_generator


# TODO: finetune model params and try different models with transfer learning
def init_model():
    '''Initializes model'''

    # params - to be finetuned
    input_shape = (256, 256, 3)
    padding = 'same'
    number_of_classes = 36
    opt = optimizers.Adam(learning_rate=0.01)

    # archtitecture
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding=padding, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding=padding))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_classes, activation='softmax'))

    model.compile(optimizer=opt,
                loss="categorical_crossentropy",
                metrics=["accuracy"])
    return model


# TODO: finetune hyperparams
def train(model, train_generator, valid_generator):
    '''Trains model on '''

    epochs = 50
    es = callbacks.EarlyStopping(patience=2,
                                 monitor='val_accuracy',
                                 restore_best_weights=True)


    model.fit_generator(generator=train_generator,
                    validation_data=valid_generator,
                    callbacks=[es],
                    verbose=1,
                    epochs=epochs)
    return model

# TODO: implement function to evaluate model on test set
'''def evaluate(model, test_generator):
        """evaluates the pipeline on df_test and return the RMSE"""
        scores = model.score(test_generator)
        accuracy = # TBD
        return accuracy'''


if __name__ == "__main__":
    df_train, df_valid = get_dataframes()
    train_generator, valid_generator = preprocess_data(df_train, df_valid, root_dir)
    model = init_model()
    trained_model = train(model, train_generator, valid_generator)