import keras
from keras.models import Sequential
from keras.layers import *


class Model:
    def createModel(self):

        model = Sequential()
        model.add(Conv2D(32, (3,3), input_shape=(32,32,3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
        model.add(Flatten())
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=26, activation='softmax'))
    
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
        model.summary()
    
    def train(self, train_generator, test_generator):
        
        past = self.fit_generator(train_generator, steps_per_epoch = 18, epoch = 3,validation_data=test_generator , validation_steps = 18)
        
        return past
        