#Import Libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#Load data set
fasion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fasion_mnist.load_data()

# View Training Image
image_index = 3
image = train_images[image_index]
print("Image Label: ", train_labels[image_index])
plt.imshow(image)

#Print shape of training images and testing images
print(train_images.shape)
print(test_images.shape)

#Create neural network model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

#Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss = 'sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#Train model
model.fit(train_images, train_labels, epochs=5, batch_size=32)

#Evaluate model
model.evaluate(test_images, test_labels)

#Make prediction
predictions = model.predict(test_images[0:5])

#Print predicted labels
print(np.argmax(predictions, axis=1))

#Print actual label values
print(test_labels[0:5])

#Print first 5 images
for i in range(0,5):
    first_image = test_images[i]
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28,28))
    plt.imshow(test_images[i])
    plt.show()
