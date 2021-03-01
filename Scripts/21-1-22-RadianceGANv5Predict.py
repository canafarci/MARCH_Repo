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


brisbane_loc = [-27.38, 88200.0]
nairobi_loc = [-1.28, 107400.0]
cairo_loc = [30.03, 63200.0]
ankara_loc = [39.90, 36100.0]
london_loc  = [51.50, 13800.0]

#load and prepare training images
def load_image(filename, size=(256,256)):
    #load image with the preferred size
    pixels = load_img(filename, target_size=size)
    #convert to an array
    pixels = img_to_array(pixels)
    pixels = convert_rgb_array_to_hsv(pixels)
    #reshape to 1 sample
    pixels = expand_dims(pixels, 0)
    return pixels

def convert_rgb_array_to_hsv(image_array):
    for i in range(0,256):
        for j in range(0,256):
            hsv = rgb_to_hsv(image_array[i, j, 0], image_array[i, j, 1], image_array[i, j, 2])
            image_array[i, j, 0] = hsv[0]
            image_array[i, j, 1] = hsv[1]
            image_array[i, j, 2] = hsv[2]
            
    image_array[:, :, 0] = (image_array[:, :, 0] - 180.0) / 180.0
    image_array[:, :, 1] = (image_array[:, :, 1] - 50.0) / 50.0
    image_array[:, :, 2] = (image_array[:, :, 2] - 50.0) / 50.0
    return image_array

#load model
model = load_model("C:\\_Thesis\VirtualEnv\\_models\\_radiancev5" + "\\12.h5", compile=False)


def plot_images(src_img, gen_img, tar_img, error):
    convert_from_hsv_to_rgb(gen_img)
    convert_from_hsv_to_rgb(tar_img)
    convert_from_hsv_to_rgb(src_img)
        
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

def convert_from_hsv_to_rgb(gen_img):
    for i in range (0, 256):
        for j in range(0, 256):
            gen_img_hsv = hsv_to_rgb(gen_img[0, i, j, 0], gen_img[0, i, j, 1], gen_img[0, i, j, 2])
            gen_img[0, i, j, 0] = gen_img_hsv[0]
            gen_img[0, i, j, 1] = gen_img_hsv[1]
            gen_img[0, i, j, 2] = gen_img_hsv[2]
    
    
def convert_loc_and_predict(image_int, longtitude):
    longtitude = asarray(longtitude)
    longtitude = expand_dims(longtitude, 0)
    longtitude = longtitude.astype("float32")
    #load source image
    imagestring = str(image_int) + ".jpg"
    src_image = load_image("C:\\_Thesis\VirtualEnv\\_datasets\\_radiancev5\\test\\source\\" + imagestring)
    tar_image = load_image("C:\\_Thesis\VirtualEnv\\_datasets\\_radiancev5\\test\\target\\" + imagestring)
    #generate from source
    gen_image = model.predict([src_image, longtitude])
    #calculate error
    ground_truth_path = "C:\\_Thesis\VirtualEnv\\_datasets\\_radiancev5\\test\\target\\" + imagestring
    ground_truth  = load_image(ground_truth_path)
    # scale from [-1,1] to [0,1]
    
    gen_image = (gen_image + 1) / 2.0
    tar_image = (tar_image + 1) / 2.0
    src_image = (src_image + 1) / 2.0
    ground_truth = (ground_truth + 1) / 2.0
    gen_image_plot = gen_image
    ground_truth = ground_truth.astype("float32")
    error = percentage_error(gen_image, ground_truth)
    #plot the image
    plot_images(src_image, gen_image_plot, tar_image, error)
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
    imageB = np.squeeze(imageB)
    imageA = imageA.astype("float32")
    imageB = imageB.astype("float32")
    #print(imageA)
    
    
    #get all white pixels
    for i in range(0,256):
        for j in range(0,256): 
            if np.array_equal(imageB[i, j], [0.0, 0.0, 1.0]):
                pass
            else:
                hue_difference = calculate_percentage(imageA, imageB, i, j)
                difference_value_list.append(hue_difference)
                k += 1
    total_hue_difference = sum(difference_value_list) / (k)
    return total_hue_difference

def calculate_percentage(imageA, imageB, i, j):
    hsv1 = imageA[i, j, 0], imageA[i, j, 1], imageA[i, j, 2]
    hsv2 = imageB[i, j, 0], imageB[i, j, 1], imageB[i, j, 2]
    h1 = hsv1[0]
    h2 = hsv2[0]
    difference = h2 - h1
    abs_difference = abs(difference)
    hue_difference = abs_difference * 100
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

def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    i = int(h*6.0) # XXX assume int() truncates!
    f = (h*6.0) - i
    p = v*(1.0 - s)
    q = v*(1.0 - s*f)
    t = v*(1.0 - s*(1.0-f))
    i = i%6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q

convert_loc_and_predict(7, 7)

""" london_loc  = [51.50, 13800.0]      5
ankara_loc = [39.90, 36100.0]           0
cairo_loc = [30.03, 63200]              2
mumbai_loc = [19.07]                    6
nairobi_loc = [-1.28, 107400]           7
lapaz_loc = [-16.48]                    4
brisbane_loc = [-27.38, 88200.0]        1
concepcion_loc = [-36.82]               3
wellington_loc                          8 
"""   

