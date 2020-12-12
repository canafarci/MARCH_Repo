import cv2
import tensorflow as tf

CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 50

def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_array = img_array/255.0
    new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("_models/2020-05-21-CatDogModel2.model")


def predict(filepath):
    if model.predict([prepare(filepath)])[0][0]*10 > 1:
        print(CATEGORIES[1], model.predict([prepare(filepath)])[0][0]*100)
    else:
        print(CATEGORIES[0], model.predict([prepare(filepath)])[0][0]*100)

predict("_predictions/dog.jpg")
predict("_predictions/cat.jpg")
predict("_predictions/cat1.jpg")
