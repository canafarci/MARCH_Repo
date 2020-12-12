import silence_tensorflow.auto
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import logging
import numpy as np
import cv2

validation_data_dir = 'D:\ders\MASTER\_Thesis\VirtualEnv\_datasets\_arcDataset\_traindata'
predict_data_dir = 'D:\ders\MASTER\_Thesis\VirtualEnv\_datasets\_arcDataset\_predict'

BATCH_SIZE = 10
IMG_SIZE = 26450


def combine_path(filename):
    return os.path.join(predict_data_dir, filename)



def predict(predictimage):
    img_array = cv2.imread(combine_path(predictimage), cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE),
                           interpolation=cv2.INTER_AREA)
    new_array = np.array(new_array).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_prob = model.predict(new_array)
    top3 = np.argsort(y_prob[0])[:-4:-1]
    print(predictimage, " = ", top3)



val_datagen = ImageDataGenerator(rescale=1. / 255)


validation_generator = val_datagen.flow_from_directory(
    predict_data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

model = tf.keras.models.load_model("_models/15-6-2020-ArchDatasetModel#8.model")


classes = list(validation_generator.class_indices.keys())

print(validation_generator.class_indices)

print(model.evaluate(validation_generator))



"""_models/15-6-2020-ArchDatasetModel#1.model -
 Adam, 10 epoch, 0.4 val split 32 32 64 64 conv + 512 128 dense """

"""_models/15-6-2020-ArchDatasetModel#2.model -
 Adamax, 15 epoch, 0.4 val split 32 32 64 64 conv + 512 128 dense |INCREASED DROPOUT| """

"""_models/15-6-2020-ArchDatasetModel#3.model -
 RMSprop, 15 epoch, 0.4 val split 32 32 64 64 conv + 512 128 dense |INCREASED DROPOUT| """

"""_models/15-6-2020-ArchDatasetModel#4.model -
 RMSprop, 15 epoch, 0.2 val split 32 32 64 64 conv + 512 128 dense |INCREASED DROPOUT| """

"""_models/15-6-2020-ArchDatasetModel#5.model -
 Adam, 15 epoch, 0.2 val split 32 32 64 conv + 128 dense + batchsize = 32 |TRUE DATA AUGMENT"""

"""_models/15-6-2020-ArchDatasetModel#6.model - 
 interpolation=cv2.INTER_CUBIC
 Adam, 20 epoch, 0.2 val split 16 32 64 conv + 128 256 dense + batchsize = 32 |TRUE DATA AUGMENT|"""

"""_models/15-6-2020-ArchDatasetModel#7.model - %12
 interpolation=cv2.INTER_CUBIC RGB IMG SIZE 150
 Adam, 20 epoch, 0.2 val split 16 32 64 conv + 128 256 dense + batchsize = 32 |TRUE DATA AUGMENT|"""

"""_models/15-6-2020-ArchDatasetModel#8.model -  %14
 interpolation=cv2.INTER_CUBIC RGB SIZE 150
 Adam, 300 epoch, 0.4 val split 16 32 64 64 conv + 128 256 dense + batchsize = 32 |TRUE DATA AUGMENT|"""

"""_models/15-6-2020-ArchDatasetModel#9.model -  %45
 interpolation=cv2.INTER_CUBIC GRAYSCALE SIZE 250
 Adam, 20 epoch, 0.2 val split 16 32 64 64 conv + 128 256 dense + batchsize = 32 |TRUE DATA AUGMENT|"""

"""_models/15-6-2020-ArchDatasetModel#10.model - 
 interpolation=cv2.INTER_CUBIC GRAYSCALE IMG SIZE 250
 Adam, 300 epoch, 0.4 val split 16 32 64 64 128 conv + 128 256 dense + batchsize = 32 |TRUE DATA AUGMENT|"""


"""_models/15-6-2020-ArchDatasetModel#12.model -  %7
 interpolation=cv2.INTER_CUBIC GRAYSCALE IMG SIZE 250
 Adam, 300 epoch, 0.4 val split 16 32 64 64 128 conv + 128 256 dense + batchsize = 32 |TRUE DATA AUGMENT|"""


"""_models/15-6-2020-ArchDatasetModel#14.model - %6
 interpolation=cv2.INTER_CUBIC GRAYSCALE IMG SIZE 125
 Adam, 150 epoch, 0.4 val split 16 32 64 64 128 conv + 128 256 dense + batchsize = 32 |TRUE DATA AUGMENT|"""

"""_models/15-6-2020-ArchDatasetModel#15.model - %8
 interpolation=cv2.INTER_CUBIC GRAYSCALE IMG SIZE 175
 Adam, 150 epoch, 0.4 val split 16 32 64 64 128 conv + 128 256 dense + batchsize = 32 |TRUE DATA AUGMENT|"""

"""_models/15-6-2020-ArchDatasetModel#16.model - %16
 interpolation=cv2.INTER_CUBIC GRAYSCALE IMG SIZE 125
 Adam, 150 epoch, 0.4 val split 16 32 64 64 128 conv + 128 256 dense + batchsize = 32 |TRUE DATA AUGMENT|"""

"""_models/15-6-2020-ArchDatasetModel#17.model - %39
 interpolation=cv2.INTER_CUBIC GRAYSCALE IMG SIZE 200
 Adam, 300 epoch, 0.4 val split 16 32 64 64 128 conv + 128 256 dense + batchsize = 64 |TRUE DATA AUGMENT|"""

"""_models/15-6-2020-ArchDatasetModel#18.model - %11
 interpolation=cv2.INTER_CUBIC GRAYSCALE IMG SIZE 140 lr 0.0005
 Adam, 300 epoch, 0.4 val split 16 32 64 64 128 conv + 128 256 dense + batchsize = 128 |TRUE DATA AUGMENT|"""

"""_models/15-6-2020-ArchDatasetModel#18.model - %23
 interpolation=cv2.INTER_CUBIC GRAYSCALE IMG SIZE 140 lr 0.0005
 Adam, 300 epoch, 0.4 val split 16 32 64 64 128 conv + 128 256 dense + batchsize = 128 |TRUE DATA AUGMENT|"""
