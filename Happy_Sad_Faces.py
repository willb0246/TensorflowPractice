import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data = #enter path here

#for when it is zipped
#change directory to whatever is appropriate (from /tmp/h-or-s)
zip_ref = zipfile.ZipFile(data, 'r')
zip_ref.extractall("/tmp/h-or-s")
zip_ref.close()

def train_happy_sad_model():
    DESIRED_ACCURACY = 0.999

    class MyCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('acc')>DESIRED_ACCURACY):
                print("\n Good Job! You got over 99.9% accuracy. Ending model training.")
                self.model.stop_training = True

    callbacks = MyCallback()

    model = tf.keras.layers.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rescale=1/255)
    train_generator = train_datagen.flow_from_directory(
        '/tmp/h-or-s',  #change this to the above directory
        target_size=(150,150),
        batch_size=10,
        class_mode='binary'
    )

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=8,
        epochs=10,
        verbose=1,
        callbacks=[callbacks]
    )

    return history.history['acc'][-1]

train_happy_sad_model()