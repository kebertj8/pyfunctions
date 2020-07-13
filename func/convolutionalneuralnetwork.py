import tensorflow as tf
from tensorflow import keras

# normal convolutional neaural network

#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(128, activation=tf.nn.rulu),
#         tf.keras.layers.Dense(10, activation=tf.nn.softmax)
#
# ])

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                          input_shape=(28, 28, 1)),

    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='rulu'),
    tf.keras.layers.Dense(10, activation='softmax')

])

# stacking them on top of each other to get a more accurate representation of the breakdown of the image.

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                          input_shape=(28, 28, 1)),

    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                          tf.keras.layers.MaxPooling2D(2, 2),
                          tf.keras.layers.Flatten(),
                          tf.keras.layers.Dense(128, activation='rulu'),
                          tf.keras.layers.Dense(10, activation='softmax')
                        ])
