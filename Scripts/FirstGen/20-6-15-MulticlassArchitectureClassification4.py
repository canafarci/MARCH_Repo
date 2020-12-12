import random
''' import silence_tensorflow.auto '''
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from PIL import Image as pillow
import pandas as pd

IMG_SIZE = 250

train_data_dir = 'D:\ders\MASTER\_Thesis\VirtualEnv\_datasets\_arcDataset\_traindata'
CATEGORIES = ['-2600_-2000', '-550_-220', '1200_1600', '1600_1700', '1720_1840',
              '1800-1900', '1890_1935', '1895_1920', '1900_1940', '1919-1965',
              '1920-1950', '1960_2000', '1980-2015', '600_800', '800_1200']

BATCH_SIZE = 32
VAL_SPLIT = 0.3

train_images = []       
train_labels = []


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(train_data_dir, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.colo)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                train_images.append(new_array)
                train_labels.append(class_num)
            except Exception as e:
                pass


create_training_data()

train_labels = pd.get_dummies(train_labels).values
train_images = np.array(train_images)

print(train_images.shape)
print(train_labels.shape)


train_features, test_features, train_targets, test_targets = train_test_split(
    train_images, train_labels,
    train_size=0.7,
    test_size=0.3,
)


print(train_features.shape)
print(train_targets.shape)
print(test_features.shape)
print(test_targets.shape)


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    zoom_range=0.2,
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

model.add(Conv2D(16, (5, 5), activation="relu", input_shape=(250, 250, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (5, 5), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (5, 5), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

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
                              epochs=300,
                              validation_data=validation_generator,
                              validation_steps=int(115*(VAL_SPLIT))
                              )

model.save("_models/15-6-2020-ArchDatasetModel#8.model")


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
