import tensorflow as tf
import matplotlib as plt

mnist = tf.keras.datasets.mnist #28x28 hadn written digits

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(x_train,y_train, epochs=3)

model.save("_models/20-5-18-numreader.model")

new_model = tf.keras.models.load_model("_models/20-5-18-numreader.model")
predictions = new_model.predict(x_test)

#show result
import numpy as np 

print(np.argmax(predictions[0]))

import matplotlib.pyplot as plt



plt.imshow(x_test[0], cmap = "binary_r")
plt.show()