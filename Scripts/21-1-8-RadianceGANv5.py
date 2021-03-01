import os
from os import listdir
import tensorflow as tf
from numpy import asarray, savez_compressed, load, zeros, ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model, Input, load_model
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, Dropout, BatchNormalization, Embedding, Reshape, Dense
from keras.preprocessing.image import img_to_array, load_img
from matplotlib import pyplot

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
"""
london_loc  = [51.50, 13800.0]
ankara_loc = [39.90, 36100.0]
cairo_loc = [30.03, 63200]
mumbai_loc = [19.07]
nairobi_loc = [-1.28, 107400]
lapaz_loc = [-16.48]
brisbane_loc = [-27.38, 88200.0]
concepcion_loc = [-36.82]
wellington_loc = [-41.28]


#Preprocess images
#load all images from directory into memory
def load_images(path, size=(256,256)):
    src_list, tar_list, lat_list = list(), list(), list()
    i = 0
    k = 0
    j = 0
    #enumerate filenames in directory, assume all are images
    for latitude in listdir(PATH):
        # dataset paths
        src_path = PATH + latitude +"/plans/"
        tar_path = PATH + latitude + "/daysim/"
        for filename in listdir(src_path):
            #load and resize the image
            pixels = load_img(src_path + filename, target_size=size)
            #convert to an array
            src_pixels = img_to_array(pixels)
            hsv_src_pixels = convert_rgb_array_to_hsv(src_pixels)
            src_list.append(hsv_src_pixels)
            print("source : " + str(j))
            j += 1
        for filename in listdir(tar_path):
            #load and resize the image
            pixels = load_img(tar_path + filename, target_size=size)
            #convert to an array
            tar_pixels = img_to_array(pixels)
            hsv_tar_pixels = convert_rgb_array_to_hsv(tar_pixels)
            tar_list.append(hsv_tar_pixels)
            lat_list.append(i)
            print("target : " +  str(k))
            k += 1
        i += 1
    lat_list = asarray(lat_list)
    lat_list = lat_list.astype("float32")
    return [asarray(src_list), asarray(tar_list), lat_list]

def convert_rgb_array_to_hsv(image_array):
    for i in range(0,256):
        for j in range(0,256):
            hsv = rgb_to_hsv(image_array[i, j, 0], image_array[i, j, 1], image_array[i, j, 2])
            image_array[i, j, 0] = hsv[0]
            image_array[i, j, 1] = hsv[1]
            image_array[i, j, 2] = hsv[2]
    return image_array

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
    v = mx * 100
    return h, s, v

#preprocess and save images
# dataset paths
PATH = "C:\_Thesis\VirtualEnv\_datasets\_radiancev5\\"
# load dataset
[src_images, tar_images, latitude] = load_images(PATH)
print("Loaded: ", src_images.shape, tar_images.shape, latitude.shape)
# save as compressed numpy array
filename = "_datasets\\_radiancev5\\rad_256.npz"
savez_compressed(filename, src_images, tar_images, latitude)
print("Saved dataset: ", filename) 
"""
"""
#load dataset
data = load("_datasets\\_radiancev4\\rad_256.npz")
src_images, tar_images, longtitude = data["arr_0"], data["arr_1"], data["arr_2"]
print("Loaded: ", src_images.shape, tar_images.shape, longtitude.shape)
#print source images
n_samples = 3
for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + i)
    pyplot.axis("off")
    pyplot.imshow(src_images[i + 530].astype("uint8"))
#plot targeted image
for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + n_samples + i)
    pyplot.axis("off")
    pyplot.imshow(tar_images[i + 530].astype("uint8"))
    pyplot.title(str(longtitude[i + 530]))

pyplot.show()


"""
####
#END OF DATAPREP
####

#define the discriminator model
def define_discriminator(image_shape, n_classes=9):
    #weight initialization
    init = RandomNormal(stddev=0.02)
    #source image input
    in_src_image = Input(shape=image_shape)
    #target image input
    in_target_image = Input(shape =image_shape)
    #longtitude label input
    longtitude = Input(shape=(1,))
    #embedding later
    em = Embedding(n_classes, 200)(longtitude)
    # scale up to image dimensions with linear activation
    n_nodes = image_shape[0] * image_shape[1]
    em = Dense(n_nodes)(em)
    # reshape to additional channel
    em = Reshape((image_shape[0], image_shape[1], 1))(em)
    #concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image, em])
    #C64
    d = Conv2D(128, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    #C128
    d = Conv2D(256, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    #C256
    d = Conv2D(512, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    #C512
    d = Conv2D(512, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    #Second last input layer
    d = Conv2D(512, (4,4), padding="same", kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    #patch output
    d = Conv2D(1, (4,4), padding="same", kernel_initializer=init)(d)
    patch_out = Activation("sigmoid")(d)
    #define model
    model = Model([in_src_image, in_target_image, longtitude], patch_out)
    #compile model
    opt = Adam(lr=0.0002, beta_1=0.75)
    model.compile(loss="binary_crossentropy", optimizer=opt, loss_weights=[0.5])
    return model

#define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    #weight initialization
    init = RandomNormal(stddev=0.02)
    #add downsampling layer
    g = Conv2D(n_filters, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(layer_in)
    #conditionally add normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    #leaky ReLU activation
    g = LeakyReLU(alpha=0.2)(g)
    return g

#define encoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    #weight initialization
    init = RandomNormal(stddev=0.02)
    #add upsampling layer
    g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(layer_in)
    #add batch normalization
    g = BatchNormalization()(g, training=True)
    #conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    #merge with skip connection
    g = Concatenate()([g, skip_in])
    #ReLU activation
    g = Activation("relu")(g)
    return g

#define standalone generator model
def define_generator(image_shape=(256, 256, 3), n_classes=9):
    #weight initialization
    init = RandomNormal(stddev=0.02)
    #longtitude label input
    longtitude = Input(shape=(1,))
    #embedding later
    em = Embedding(n_classes, 200)(longtitude)
    # scale up to image dimensions with linear activation
    n_nodes = image_shape[0] * image_shape[1]
    em = Dense(n_nodes)(em)
    # reshape to additional channel
    em = Reshape((image_shape[0], image_shape[1], 1))(em)
    #image input
    in_image = Input(shape=image_shape)
    #concentanate inputs
    merged = Concatenate()([in_image, em])
    #encoder model
    e1 = define_encoder_block(merged, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 521)
    #bottleneck, no batch norm and relu
    b = Conv2D(512, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(e7)
    b = Activation("relu")(b)
    #decoder model
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    #output
    g = Conv2DTranspose(3, (4,4), strides =(2,2), padding="same", kernel_initializer=init)(d7)
    out_image = Activation("tanh")(g)
    #define model
    model = Model([in_image, longtitude], out_image)
    return model

def define_gan(g_model, d_model, image_shape):
    #make weights in the discriminator not trainable
    d_model.trainable = False
    #define source image
    in_src = Input(shape=image_shape)
    #define source longtitude
    longtitude = Input(shape=(1,))
    #connect the source image to the generator input
    gen_out = g_model([in_src, longtitude])
    #connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out, longtitude])
    #src image as input, generated image and classification output
    model = Model([in_src, longtitude], [dis_out, gen_out])
    #compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=["binary_crossentropy", "mae", "binary_crossentropy"], optimizer=opt, loss_weights=[1,100, 1])
    return model

#load and prepare training images
def load_real_samples(filename):
    #load the compressed arrays
    data = load(filename)
    #unpack the arrays
    X1, X2, X3 = data["arr_0"], data["arr_1"], data["arr_2"]
    #scale from hsv to [-1,1]
    X1[:, :, :, 0] = (X1[:, :, :, 0] - 180.0) / 180.0
    X1[:, :, :, 1] = (X1[:, :, :, 1] - 50.0) / 50.0
    X1[:, :, :, 2] = (X1[:, :, :, 2] - 50.0) / 50.0
    
    X2[:, :, :, 0] = (X2[:, :, :, 0] - 180.0) / 180.0
    X2[:, :, :, 1] = (X2[:, :, :, 1] - 50.0) / 50.0
    X2[:, :, :, 2] = (X2[:, :, :, 2] - 50.0) / 50.0
    return[X1, X2, X3]

#select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    #unpack dataset
    trainA, trainB, trainC = dataset
    #choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    #retrieve selected images
    X1, X2, X3 = trainA[ix], trainB[ix], trainC[ix]
    #generate "real" class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2, X3], y

#generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape, longtitude):
    #generate fake instance
    X = g_model.predict([samples, longtitude])
    #create "fake" class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

####
#END OF GAN DEFINITION
####

#generate samples and save as a plot and save the model
def summarize_performance(step, g_model, d_model, gan_model, dataset, n_samples=3):
    #select a sample of input images
    [X_realA, X_realB, X_realC], _ = generate_real_samples(dataset, n_samples, 1)
    #generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1, X_realC)
    #scale all pixels from [-1,1] to [0,1] and convert to rgb
    X_realA, X_fakeB, X_realB = convert_and_normalize_hsv_to_rgb(X_realA, X_realB, X_fakeB)
    #plot real source images
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis("off")
        pyplot.imshow(X_realA[i])
    #plot generated target iamge
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis("off")
        pyplot.imshow(X_fakeB[i])
    #plot reak target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
        pyplot.axis("off")
        pyplot.imshow(X_realB[i])
        pyplot.title(str(X_realC[i]))
    #save plot to file
    filename1 = "__ganResults\\radiancev5\\plot_%06d.png" % (step + 1)
    pyplot.savefig(filename1)
    pyplot.close()
    #save the generator model
    filename2 = "_models\\_radiancev5\\model%06d.h5" % (step + 1)
    g_model.save(filename2)
    #save the disc model
    filename3 = "_models\\_radiancev5\\disc.h5"
    d_model.save(filename3)
    #save the gan model
    filename4 = "_models\\_radiancev5\\gan.h5"
    gan_model.save(filename4)
    print(">Saved: %s and %s" % (filename1, filename2))

def convert_and_normalize_hsv_to_rgb(X_realA, X_realB, X_fakeB):
    #scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    for i in range (0, 3):
        for j in range(0, 256):
            for k in range(0, 256):
                X_realA_hsv = hsv_to_rgb(X_realA[i, j, k, 0], X_realA[i, j, k, 1], X_realA[i, j, k, 2])
                #print(X_realA_hsv)
                X_realA[i, j, k, 0] = X_realA_hsv[0]
                X_realA[i, j, k, 1] = X_realA_hsv[1]
                X_realA[i, j, k, 2] = X_realA_hsv[2]
                X_realB_hsv = hsv_to_rgb(X_realB[i, j, k, 0], X_realB[i, j, k, 1], X_realB[i, j, k, 2])
                X_realB[i, j, k, 0] = X_realB_hsv[0]
                X_realB[i, j, k, 1] = X_realB_hsv[1]
                X_realB[i, j, k, 2] = X_realB_hsv[2]
                X_fakeB_hsv = hsv_to_rgb(X_fakeB[i, j, k, 0], X_fakeB[i, j, k, 1], X_fakeB[i, j, k, 2])
                X_fakeB[i, j, k, 0] = X_fakeB_hsv[0]
                X_fakeB[i, j, k, 1] = X_fakeB_hsv[1]
                X_fakeB[i, j, k, 2] = X_fakeB_hsv[2]
    return X_realA, X_fakeB, X_realB

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
    
#####
#END OF Save photos incrementally
#####

#train pix2pix model
def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
    #determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    #unpack dataset
    trainA, trainB, trainC = dataset
    #calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    #calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    #manually enumerate epochs
    for i in range(n_steps):
        #select a batch of real samples
        [X_realA, X_realB, X_realC], y_real = generate_real_samples(dataset, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch, X_realC)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB, X_realC], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB, X_realC], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch([X_realA, X_realC], [y_real, X_realB])
        # summarize performance
        print(">%d, d1[%.3f] d2[%.3f] g[%.3f]" % (i+1, d_loss1, d_loss2, g_loss))
        # summarize model performance
        if (i+1) % (bat_per_epo) == 0:
            summarize_performance(i, g_model, d_model, gan_model, dataset)
            
####
####END OF DEFINITIONS
####

# load image data
dataset = load_real_samples("_datasets\\_radiancev5\\rad_256.npz")
print("Loaded ", dataset[0].shape, dataset[1].shape, dataset[2].shape)

# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)
# train model
train(d_model, g_model, gan_model, dataset)
#load model

""" d_model = load_model("_models\\_radiancev5\\disc.h5")
g_model = load_model("_models\\_radiancev5\\7.h5")
# define the composite model
gan_model = load_model("_models\\_radiancev5\\gan.h5")
# train model
train(d_model, g_model, gan_model, dataset) """
