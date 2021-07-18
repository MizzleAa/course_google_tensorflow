'''
Below is code with a link to a happy or sad dataset which contains 80 images, 40 happy and 40 sad. Create a convolutional neural network that trains to 100% accuracy on these images, which cancels training upon hitting training accuracy of >.999

Hint -- it will work best with 3 convolutional layers.

file link : https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip
'''

import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = f"./tmp/happy-or-sad.zip"

zip_ref = zipfile.ZipFile(path, 'r')
zip_ref.extractall("/tmp/h-or-s")
zip_ref.close()

def train_happy_sad_model():
    
    class myCallback(tf.keras.callbacks.Callback):
         def on_epoch_end(self, callback, logs={}):
                if (logs.get('acc') > 0.999):
                    print("\nReached 0.999% accuracy so cancelling training!")
                    self.model.stop_training = True

    callbacks = myCallback()
    
    model = tf.keras.models.Sequential([
        # Your Code Here
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150,150,3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['acc'])

    train_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        '/tmp/h-or-s',
        target_size=(150, 150),
        batch_size=10,
        class_mode='binary'
    )

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=8,
        epochs=15,
        verbose=1,
        callbacks=[callbacks]
    )
    # model fitting
    return history.history['acc'][-1]

train_happy_sad_model()