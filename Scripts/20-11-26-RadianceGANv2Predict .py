from keras.models import load_model
from numpy import expand_dims, vstack, asarray
from matplotlib import pyplot
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
import numpy as np

#load and prepare training images
def load_image(filename, size=(256,256)):
    #load image with the preferred size
    pixels = load_img(filename, target_size=size)
    #convert to an array
    pixels = img_to_array(pixels)
    #scale from [0,255] to [-1,1]
    pixels = (pixels - 127.5) / 127.5
    #reshape to 1 sample
    pixels = expand_dims(pixels, 0)
    return pixels

#load model
model = load_model("C:\\_Thesis\VirtualEnv\\_models\\_radiancev2" + "\\model8.h5")

def plot_images(src_img, gen_img, tar_img):
    images = vstack((src_img, gen_img, tar_img))
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    titles = ["Source", "Generated", "Expected"]
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        pyplot.subplot(1, 3, 1 + i)
        # turn off axis
        pyplot.axis("off")
        # plot raw pixel data
        pyplot.imshow(images[i])
        # show title
        pyplot.title(titles[i])
    pyplot.show()
    
def convert_lon_and_predict(image_int, longtitude):
    longtitude = asarray(longtitude)
    longtitude = expand_dims(longtitude, 0)
    longtitude = longtitude.astype("float32")
    #load source image
    imagestring = str(image_int) + ".bmp"
    src_image = load_image("C:\\_Thesis\VirtualEnv\\_datasets\\_radiancev2\\test\\source\\" + imagestring)
    tar_image = load_image("C:\\_Thesis\VirtualEnv\\_datasets\\_radiancev2\\test\\target\\" + imagestring)
    #generate from source
    gen_image = model.predict([src_image, longtitude])
    #plot the image
    plot_images(src_image, gen_image, tar_image)

convert_lon_and_predict(5, 0)