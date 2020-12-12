import silence_tensorflow.auto
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt



IMG_SIZE = 150


train_data_dir = 'D:\ders\MASTER\_Thesis\VirtualEnv\_datasets\_arcDataset\_traindata'
validation_data_dir = 'D:\ders\MASTER\_Thesis\VirtualEnv\_datasets\_arcDataset\_traindata'

BATCH_SIZE = 16
VAL_SPLIT = 0.2

datagen = ImageDataGenerator(
                            rescale=1. / 255,
                            rotation_range=20,
                            zoom_range=0.2,
                            width_shift_range=0.125,
                            height_shift_range=0.125,
                            horizontal_flip=True,
                            validation_split=VAL_SPLIT
                            )

train_generator = datagen.flow_from_directory(
                            train_data_dir,
                            target_size=(IMG_SIZE,IMG_SIZE),
                            batch_size=BATCH_SIZE,
                            subset="training",
                            class_mode="categorical"
                            )

val_datagen = ImageDataGenerator(rescale= 1. / 255)

validation_generator = datagen.flow_from_directory(
                            validation_data_dir,
                            target_size=(IMG_SIZE,IMG_SIZE),
                            batch_size=BATCH_SIZE,
                            subset="validation",
                            class_mode="categorical"
                            )

model = Sequential()

model.add(Conv2D(32, kernel_size=5, activation="relu", input_shape=(150,150,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(32, kernel_size=5, activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, kernel_size=5, activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size=5, activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.1))

model.add(Dense(15, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001)

model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=["categorical_accuracy"])

history = model.fit_generator(train_generator,
                              steps_per_epoch=230 - int(230*(VAL_SPLIT)),
                              epochs= 15,
                              validation_data=validation_generator,
                              validation_steps=int(230*(VAL_SPLIT))
                              )

model.save("_models/15-6-2020-ArchDatasetModel#5.model")













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
