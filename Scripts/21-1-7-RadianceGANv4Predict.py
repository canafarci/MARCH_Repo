from keras.models import load_model
from numpy import expand_dims, vstack, asarray, count_nonzero, sum, squeeze, hstack
from matplotlib import pyplot
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
import numpy as np
from cv2 import cv2
from skimage import metrics

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#### LAT, GHI LIST
london_loc  = [51.50, 13800.0]
ankara_loc = [39.90, 36100.0]
brisbane_loc = [-27.38, 88200.0]
nairobi_loc = [-1.28, 107400.0]
cairo_loc = [30.03, 63200.0]

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
model = load_model("C:\\_Thesis\VirtualEnv\\_models\\_radiancev4" + "\\2.h5", compile=False)

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
    
def convert_loc_and_predict(image_int, location):
    #normalize data
    lat_list = ((float(location[0])-(-27.38)))/((51.50)-(-27.38))
    ghi_list = ((float(location[1])-(13800.0)))/((107400.0)-(13800.0))
    #convert to numpy array
    lat_list = asarray(lat_list)
    lat_list = lat_list.astype("float32")
    ghi_list = asarray(ghi_list)
    ghi_list = ghi_list.astype("float32")
    #stack arrays
    #lat_list = expand_dims(lat_list, axis=0)
    #ghi_list = expand_dims(ghi_list, axis=0)
    loc_array = hstack((lat_list, ghi_list))
    loc_array = 2 * (loc_array - 0.5)
    loc_array = asarray(loc_array)
    loc_array = expand_dims(loc_array, axis=0)
    #load source image
    imagestring = str(image_int) + ".png"
    src_image = load_image("C:\\_Thesis\VirtualEnv\\_datasets\\_radiancev4\\test\\source\\" + imagestring)
    tar_image = load_image("C:\\_Thesis\VirtualEnv\\_datasets\\_radiancev4\\test\\target\\" + imagestring)
    #generate from source
    gen_image = model.predict([src_image, loc_array])
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
                hue_difference = calculate_percentage(imageA, imageB, i, j)
                difference_value_list.append(hue_difference)
                k += 1
    total_hue_difference = sum(difference_value_list) / (k)
    return total_hue_difference

def calculate_percentage(imageA, imageB, i, j):
    print(imageB[i, j])
    print(imageA[i, j])
    hsv1 = rgb_to_hsv(imageA[i, j, 0], imageA[i, j, 1], imageA[i, j, 2])
    hsv2 = rgb_to_hsv(imageB[i, j, 0], imageB[i, j, 1], imageB[i, j, 2])
    print(hsv1)
    print(hsv2)
    h1 = hsv1[0]
    h2 = hsv2[0]
    difference = h2 - h1
    abs_difference = abs(difference)
    hue_difference = abs_difference / 360.0 * 100
    return hue_difference
    
def compare_images(imageA, imageB):
    # compute the mean squared error and structural similarity
    # index for the images
    print(imageA.shape)
    print(imageA.shape)
    m = mse(imageA, imageB)
    s = metrics.structural_similarity(imageA, imageB)
    return m, s

def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h, s, v

convert_loc_and_predict(2, ankara_loc)

