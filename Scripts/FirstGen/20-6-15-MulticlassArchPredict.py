import silence_tensorflow.auto
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os, logging
import numpy as np

validation_data_dir = 'D:\ders\MASTER\_Thesis\VirtualEnv\_datasets\_arcDataset\_traindata'
predict_data_dir = 'D:\ders\MASTER\_Thesis\VirtualEnv\_datasets\_arcDataset\_predict'

BATCH_SIZE = 16
IMG_SIZE = 250

def combine_path(filename):
    return os.path.join(predict_data_dir, filename)

def predict(predictimage):
    img = image.load_img(combine_path(predictimage), target_size=(IMG_SIZE, IMG_SIZE, 3))
    img = image.img_to_array(img)
    img = img/255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)
    y_prob = model.predict(img)
    top3 = np.argsort(y_prob[0])[:-4:-1]
    print(predictimage, " = ", top3)

val_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="validation",
    class_mode="categorical"
)

model = tf.keras.models.load_model("_models/15-6-2020-ArchDatasetModel#8.model")



classes = list(validation_generator.class_indices.keys())

print(validation_generator.class_indices)
predict("1.jpg")
predict("2.jpg")
predict("3.jpg")
predict("4.jpg")
predict("5.jpg")
predict("6.jpg")
predict("7.jpg")
predict("8.jpg")
predict("9.jpg")
predict("10.jpg")


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
