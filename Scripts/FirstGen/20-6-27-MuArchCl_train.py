import random
''' import silence_tensorflow.auto '''
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import pickle

IMG_SIZE = 175
BATCH_SIZE = 32
VAL_SPLIT = 0.4

train_images = pickle.load(open("_pickle\X_arcDataset.pickle", "rb"))
train_labels = pickle.load(open("_pickle\y_arcDataset.pickle", "rb"))

train_images = train_images/255.0


train_features, test_features, train_targets, test_targets = train_test_split(
    train_images, train_labels,
    train_size=1 - VAL_SPLIT,
    test_size=VAL_SPLIT,
    shuffle=True,
)


print(train_features.shape)
print(train_targets.shape)
print(test_features.shape)
print(test_targets.shape)


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)


train_generator = train_datagen.flow(
    train_features,
    train_targets,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_datagen = ImageDataGenerator(
    rescale=1. / 255,
)

validation_generator = validation_datagen.flow(
    test_features,
    test_targets,
    batch_size=BATCH_SIZE,
    shuffle=True
)



model = Sequential()


model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))


model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))


model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(15, activation="softmax"))


opt = tf.keras.optimizers.Adam(lr=0.001)


model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=["categorical_accuracy"])

history = model.fit_generator(train_generator,
                              steps_per_epoch=115 - int(115. * (VAL_SPLIT)),
                              epochs=100,
                              validation_data=validation_generator,
                              validation_steps=int(115*(VAL_SPLIT))
                              )

model.save("_models/27-6-2020-ArchDatasetModel#15.model")


##########################   TRAINING VISUALIZATION   ############################

loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

##########################   TRAINING VISUALIZATION   ############################
