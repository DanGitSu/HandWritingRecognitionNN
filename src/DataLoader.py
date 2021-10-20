from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

class DataLoader:

    def createData(self, rescale, shear_range, zoom_range, horizontal_flip):

        train_datagen = ImageDataGenerator(rescale, shear_range, zoom_range, horizontal_flip)

        test_datagen = ImageDataGenerator(rescale)

        train_generator = train_datagen.flow_from_directory(
            directory='data/Training',
            target_size=(32, 32),
            batch_size=32,
            class_mode='categorical'

        )

        test_generator = test_datagen.flow_from_directory(
            directory='data/Testing',
            target_size=(32, 32),
            batch_size=32,
            class_mode='categorical'
        )

        output_array = {train_generator, test_generator}

        return output_array