import keras
from keras.models import Sequential
from keras.layers import *


class Model:

    model = Sequential()
    def createModel(self):

        self.model.add(Conv2D(32, (3, 3), input_shape=(32,32,3), activation = 'relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
    
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
    
        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dense(units=26, activation='softmax'))
    
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    
        self.model.summary()

    #training is shit because its not done
    #need to be able to validate and test the model