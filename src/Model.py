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

    def train(self, train_generator, test_generator):
        past = self.model.fit(train_generator, steps_per_epoch=18, epochs=3, validation_data=test_generator,
                         validation_steps=18)


    # def evaluate(self,train_generator, test_generator):
    #     scores = self.model.evaluate(train_generator, test_generator, verbose=0)
    #     print("CNN Error: %.2f%%" % (100 - scores[1] * 100))

    #training is shit because its not done
    #need to be able to validate and test the model