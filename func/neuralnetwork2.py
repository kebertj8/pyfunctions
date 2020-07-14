import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images,
                            test_labels) = fashion_mnist.load_data()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)

])

model.compile(optimizer=tf.train.AdamOptimizer(),
            loss='sparse_categorical_crossentropy')

# loss calculates how bad the neural network works and then
#  the optimizer runs the program again to get it even better


# fit the training images to the training labels

model.fit(train_images, train_labels, epochs=5)

# testing the model with images it hasnt previoulsy seen and see how it performs

test_loss, test_acc = model.evaluate(test_images, test_labels)

#  then we get call to see if it recongizes the images.

predictions = model.predict(my_images)

# basically printing the function
