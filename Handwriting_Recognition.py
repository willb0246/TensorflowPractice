import tensorflow as tf
from os import path, getcwd, chdir

data = #import handwriting data here

def  train_mnist():

    class MyCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('acc')>0.99):
                print("\n Good Job! You reached 99% accuracy so were stopping training.")
                self.model.stop_training = True

    callbacks = MyCallback()

    minst = tf.keras.datasets.minst

    (pic_train, label_train), (pic_test, label_test) = mnist.load_data(path=data)

    pic_train = pic_train / 255.0
    pic_test = pic_test / 255.0

    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(128, activation='relu'),
                                        tf.keras.layers.Dense(10, activation='softmax')])

    model.compile(optimizer = 'adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(pic_test, label_test, epochs=10, callbacks=[callbacks])

    return history.eopch, history.history['acc'][-1]

train_mnist()