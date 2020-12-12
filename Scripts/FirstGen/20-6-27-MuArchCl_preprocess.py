import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd


IMG_SIZE = 64

train_data_dir = 'C:/_Thesis/VirtualEnv/_datasets/_arcDataset/_traindata'
CATEGORIES = ['-2600_-2000', '1200_1600', '1600_1700', '1720_1840',
              '1800-1900', '1890_1935', '1895_1920', '1900_1940', '1919-1965', '1919-1965-2',
              '1920-1950', '1960_2000', '1980-2015', '600_800', '800_1200']

BATCH_SIZE = 32
VAL_SPLIT = 0.4

train_images = []
train_labels = []
training_data = []


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))


def create_training_data():
    counter = 0
    for category in CATEGORIES:
        path = os.path.join(train_data_dir, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                im = Image.open(os.path.join(path, img))
                ima = crop_max_square(im)
                imag = ima.convert("L")
                image = imag.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                np_image = np.array(image)
                training_data.append([np_image, class_num])
                counter += 1
                print(counter)
            except Exception as e:
                print("err")
                pass
            
create_training_data()


random.shuffle(training_data)

for features, label in training_data:
    train_images.append(features)
    train_labels.append(label)

train_labels = pd.get_dummies(train_labels).values
train_images = np.array(train_images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

import pickle

pickle_out = open("_pickle\X_arcDataset.pickle", "wb")
pickle.dump(train_images, pickle_out)
pickle_out.close()

pickle_out = open("_pickle\y_arcDataset.pickle", "wb")
pickle.dump(train_labels, pickle_out)
pickle_out.close()

