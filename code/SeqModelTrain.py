import cv2, os
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

size = 254

Train_Directory = 'NewDataset/Train/'
Test_Directory = 'NewDataset/Test/'
Valid_Directory = 'NewDataset/Valid/'


train_images = ImageDataGenerator().flow_from_directory(Train_Directory, target_size=(size, size),
                                                        classes=['A', 'B', 'C', 'D', 'DELETE', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'SPACE', 'T', 'U', 'V', 'W', 'X', 'Y'],
                                                        color_mode='grayscale', batch_size=25)

test_images = ImageDataGenerator().flow_from_directory(Test_Directory, target_size=(size, size),
                                                        classes=['A', 'B', 'C', 'D', 'DELETE', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'SPACE', 'T', 'U', 'V', 'W', 'X', 'Y'],
                                                        color_mode='grayscale', batch_size=5)

valid_images = ImageDataGenerator().flow_from_directory(Valid_Directory, target_size=(size, size),
                                                        classes=['A', 'B', 'C', 'D', 'DELETE', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'SPACE', 'T', 'U', 'V', 'W', 'X', 'Y'],
                                                        color_mode='grayscale', batch_size=5)


def create_model():

    model = Sequential()
    model.add(Conv2D(16, (2, 2), input_shape=(size, size, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Conv2D(128, (7, 7), activation='relu'))
    model.add(MaxPooling2D(pool_size=(7, 7), strides=(7, 7), padding='same'))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(27, activation='softmax'))

    sgd = optimizers.SGD(lr=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


def train_model():
    model = create_model()
    # model.summary()
    model.fit_generator(train_images, steps_per_epoch=20,
                        validation_data=valid_images, validation_steps=10, epochs=20, verbose=2)
    model.save('SeqModel.h5')


# train_model()
