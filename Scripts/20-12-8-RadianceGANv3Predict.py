from keras.models import load_model
from numpy import expand_dims, vstack, asarray, count_nonzero, sum, squeeze
from matplotlib import pyplot
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
import numpy as np
from cv2 import cv2
from skimage import metrics

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


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
model = load_model("C:\\_Thesis\VirtualEnv\\_models\\_radiancev3" + "\\50.h5")

def plot_images(src_img, gen_img, tar_img, error):
    images = vstack((src_img, gen_img, tar_img))
    
    titles = ["Source", "Generated", "Expected"]
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        pyplot.subplot(1, 3, 1 + i)
        # turn off axis
        pyplot.axis("off")
        # plot raw pixel data
        pyplot.imshow(images[i])
        if i == 1:
             # show title
            pyplot.title("%s  \n percentage difference :  %.2f " % (titles[i], error))
        else:
            # show title
            pyplot.title(titles[i])
    pyplot.show()
    
def convert_lon_and_predict(image_int, longtitude):
    longtitude = asarray(longtitude)
    longtitude = expand_dims(longtitude, 0)
    longtitude = longtitude.astype("float32")
    #load source image
    imagestring = str(image_int) + ".png"
    src_image = load_image("C:\\_Thesis\VirtualEnv\\_datasets\\_radiancev3\\test\\source\\" + imagestring)
    tar_image = load_image("C:\\_Thesis\VirtualEnv\\_datasets\\_radiancev3\\test\\target\\" + imagestring)
    #generate from source
    gen_image = model.predict([src_image, longtitude])
    #calculate error
    ground_truth_path = "C:\\_Thesis\VirtualEnv\\_datasets\\_radiancev3\\test\\target\\" + imagestring
    ground_truth  = cv2.imread(ground_truth_path, 1)
    ground_truth = cv2.resize(ground_truth, (256, 256))
    ground_truth = asarray(ground_truth).reshape(256, 256, 3)
    ground_truth = (ground_truth - 127.5) / 127.5
    # scale from [-1,1] to [0,1]
    gen_image = (gen_image + 1) / 2.0
    tar_image = (tar_image + 1) / 2.0
    src_image = (src_image + 1) / 2.0
    ground_truth = (ground_truth + 1) / 2.0
    ground_truth = ground_truth.astype("float32")
    error = percentage_error(gen_image, ground_truth)
    #plot the image
    plot_images(src_image, gen_image, tar_image, error)
    print(gen_image.dtype, "  ", ground_truth.dtype)

def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def percentage_error(imageA, imageB):
    k = 0
    difference_value_list = list()
    imageA = np.squeeze(imageA)
    imageA = (imageA * 127.5) + 127.5
    imageB = (imageB * 127.5) + 127.5
    imageA = imageA.astype("float32")
    imageB = imageB.astype("float32")
    print(imageA.shape)
    print(imageB.shape)
    """ difference = imageA.astype("float") - imageB.astype("float")
    abs_difference = np.abs(difference)
    percentage_difference = abs_difference / 255.0
    total_percentage_difference = np.sum(percentage_difference) / (256 * 256) """
    #get all white pixels
    for i in range(0,255):
        for j in range(0,255): 
            if np.array_equal(imageB[i, j], [255.0, 255.0, 255.0]):
                pass
            else:
                print(imageB[i, j])
                print(imageA[i, j])
                difference = imageA[i, j] - imageB[i, j]
                abs_difference = np.abs(difference)
                percentage_difference = abs_difference / 255.0 * 100
                total_pixel_difference = np.sum(percentage_difference) / (3)
                difference_value_list.append(total_pixel_difference)
                k += 1
    total_percentage_difference = sum(difference_value_list) / (k)
    return total_percentage_difference

def compare_images(imageA, imageB):
    # compute the mean squared error and structural similarity
    # index for the images
    print(imageA.shape)
    print(imageA.shape)
    m = mse(imageA, imageB)
    s = metrics.structural_similarity(imageA, imageB)
    return m, s

convert_lon_and_predict(4, 3)