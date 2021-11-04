from Model import Model
from DataLoader import DataLoader
import pickle
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import *
import os
import matplotlib.pyplot as plt
import numpy as np

# Could add Validation mode later
mode = "TRAIN" # "TRAIN" or "VALIDATE"



class FilePaths:
    "filenames and paths for data"
    fnTrain = '' # to once training data can be found add here.

def get_result(result):
    if result[0][0] == 1:
        return ('a')
    elif result[0][1] == 1:
        return ('b')
    elif result[0][2] == 1:
        return ('c')
    elif result[0][3] == 1:
        return ('d')
    elif result[0][4] == 1:
        return ('e')
    elif result[0][5] == 1:
        return ('f')
    elif result[0][6] == 1:
        return ('g')
    elif result[0][7] == 1:
        return ('h')
    elif result[0][8] == 1:
        return ('i')
    elif result[0][9] == 1:
        return ('j')
    elif result[0][10] == 1:
        return ('k')
    elif result[0][11] == 1:
        return ('l')
    elif result[0][12] == 1:
        return ('m')
    elif result[0][13] == 1:
        return ('n')
    elif result[0][14] == 1:
        return ('o')
    elif result[0][15] == 1:
        return ('p')
    elif result[0][16] == 1:
        return ('q')
    elif result[0][17] == 1:
        return ('r')
    elif result[0][18] == 1:
        return ('s')
    elif result[0][19] == 1:
        return ('t')
    elif result[0][20] == 1:
        return ('u')
    elif result[0][21] == 1:
        return ('v')
    elif result[0][22] == 1:
        return ('w')
    elif result[0][23] == 1:
        return ('x')
    elif result[0][24] == 1:
        return ('y')
    elif result[0][25] == 1:
        return ('z')

def main():
    "this is the main function it will kick everything off"

    rescale = 1. / 255
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True


    if mode == "TRAIN":

        train_datagen = ImageDataGenerator(rescale, shear_range, zoom_range, horizontal_flip)

        test_datagen = ImageDataGenerator(rescale)

        train_generator = train_datagen.flow_from_directory(
            directory='../data/Training',
            target_size=(32, 32),
            batch_size=32,
            class_mode='categorical'

        )

        test_generator = test_datagen.flow_from_directory(
            directory='../data/Testing',
            target_size=(32, 32),
            batch_size=32,
            class_mode='categorical'
        )
        model = Model()
        model.createModel()

        # pickle.dump(model, open('CNN.sav', 'wb'))
        # model = pickle.load(open('CNN.sav', 'rb'))

        filename = r'../data/Testing\a\21.png'
        test_image = image.load_img(filename, target_size=(32, 32))
        # plt.imshow(test_image)
        #test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        result = model.result(test_image)
        r = get_result(result)
        print('Predicted Alphabet is: ', r)




print("Running")
main()
print("fit worked")