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
from tensorflow.keras.preprocessing import image
import imageio

im_count = 3680

IMG_SIZE = 200


train_data_dir = 'D:\ders\MASTER\_Thesis\VirtualEnv\_datasets\_arcDataset\_traindata'
CATEGORIES = ['-2600_-2000', '-550_-220', '1200_1600', '1600_1700', '1720_1840',
              '1800-1900', '1890_1935', '1895_1920', '1900_1940', '1919-1965',
              '1920-1950', '1960_2000', '1980-2015', '600_800', '800_1200']

BATCH_SIZE = 48
VAL_SPLIT = 0.4

train_images = []
train_labels = []

ba_count = im_count // BATCH_SIZE


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(train_data_dir, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(
                    path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(
                    img_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                train_images.append(new_array)
                train_labels.append(class_num)
            except Exception as e:
                pass


create_training_data()

train_labels = pd.get_dummies(train_labels).values
train_images = np.array(train_images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print(train_images.shape)
print(train_labels.shape)


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
    rotation_range=20,
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

fnames = [os.path.join(train_data_dir, "1980-2015", fname) for
          fname in os.listdir(os.path.join(train_data_dir, "1980-2015"))]




anim_file = 'prerprocess.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  print(len(fnames))
  for h in range(len(fnames)):

    img_path = fnames[h]
    img = image.load_img(img_path, target_size=(150, 150))          
    x = image.img_to_array(img)                                     
    x = x.reshape((1,) + x.shape)                                   
    i = 0   
    for batch in train_datagen.flow(x, batch_size=1):                                                       
        i += 1                                                     
        if i % 5 == 0:
            break
        writer.append_data(batch[0])
    writer.append_data(batch[0])


