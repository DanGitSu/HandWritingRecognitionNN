from Model import Model
from DataLoader import DataLoader
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import os

# Could add Validation mode later
mode = "TRAIN" # "TRAIN" or "VALIDATE"



class FilePaths:
    "filenames and paths for data"
    fnTrain = '' # to once training data can be found add here.
    
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
        past = model.train(train_generator,test_generator)
        pickle.dump(model, open('CNN.sav', 'wb'))
        model = pickle.load(open('CNN.sav', 'rb'))

        
print("Running")
main()