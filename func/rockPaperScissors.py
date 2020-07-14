# import the files

import zipfile
import os
!wget - -no - check - certificate \
    https: // storage.googleapis.com / laurencemoroney - blog.appspot.com / rps.zip \
    - O / tmp / rps.zip

!wget - -no - check - certificate \
    https: // storage.googleapis.com / laurencemoroney - blog.appspot.com / rps - test - set.zip \
    - O / tmp / rps - test - set.zip

# pythons shortcut algorithm to unzip the files internally and dispatching them into appropriate folders.


local_zip = '/tmp/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/')
zip_ref.close()

local_zip = '/tmp/rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/')
zip_ref.close()

# creates image data generator that generates images for the training

TRAINING_DIR = "/tmp/rps/"
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150, 150),
    class_mode='categorical',
)


validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
  target_size=(150, 150),
    class_mode='categorical',
    batch_size=126
)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                          input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(train_generator, epochs=25, steps_per_epoch=20,
                    validation_data=validation_generator,
                    verbose=1, validation_steps=3)

model.save("rps.h5")

classes = model.predict(images, batch_size=10)
