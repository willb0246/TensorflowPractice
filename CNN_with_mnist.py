import tensorflow as tf
from os import path, getcwd, chdir

#data = put path here

def train_cnn_w_mnist():
    class MyCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('acc')>0.9):
                print("\n Good Job! You got over 99.8% accuracy, so ending training.")
                self.model.stop_training = True

    callbacks = MyCallback()

    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data(path=data)

    training_images = training_images / 255.0
    test_images = test_images / 255.0

    model = tf.keras.layers.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizers='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(training_images, training_labels, epochs=20, callbacks=[callbacks])
    return history.epoch, history.history['acc'][-1]