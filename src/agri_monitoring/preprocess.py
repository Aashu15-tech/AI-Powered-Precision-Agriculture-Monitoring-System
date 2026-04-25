import tensorflow as tf
from keras import layers, models

def get_augmentation():
    return tf.keras.Sequential([
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])

