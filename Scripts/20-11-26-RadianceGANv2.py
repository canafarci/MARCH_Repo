import os
from os import listdir
import tensorflow as tf
from numpy import asarray, savez_compressed, load, zeros, ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model, Input
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, Dropout, BatchNormalization
from keras.preprocessing.image import img_to_array, load_img
from matplotlib import pyplot

PATH = "C:\_Thesis\VirtualEnv\_datasets\_radiance1"
print (PATH)
 
#Preprocess images
#load all images from directory into memory
def load_images(src_path, tar_path, size=(256,256)):
    src_list, tar_list = list(), list()
    #enumerate filenames in directory, assume all are images
    for filename in listdir(src_path):
        #load and resize the image
        pixels = load_img(src_path + filename, target_size=size)
        #convert to an array
        src_pixels = img_to_array(pixels)
        src_list.append(src_pixels)
    for filename in listdir(tar_path):
        #load and resize the image
        pixels = load_img(tar_path + filename, target_size=size)
        #convert to an array
        tar_pixels = img_to_array(pixels)
        tar_list.append(tar_pixels)
    return [asarray(src_list), asarray(tar_list)]

"""
#preprocess and save images
# dataset paths
src_path = PATH +"/plans/"
tar_path = PATH +"/daysim/"
# load dataset
[src_images, tar_images] = load_images(src_path, tar_path)
print("Loaded: ", src_images.shape, tar_images.shape)
# save as compressed numpy array
filename = "_datasets\\_radiance1\\rad_256.npz"
savez_compressed(filename, src_images, tar_images)
print("Saved dataset: ", filename) 
"""

#load dataset
data = load("_datasets\\_radiance1\\rad_256.npz")
src_images, tar_images = data["arr_0"], data["arr_1"]
print("Loaded: ", src_images.shape, tar_images.shape)
#print source images
n_samples = 3
for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + i)
    pyplot.axis("off")
    pyplot.imshow(src_images[i].astype("uint8"))
#plot targeted image
for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + n_samples + i)
    pyplot.axis("off")
    pyplot.imshow(tar_images[i].astype("uint8"))

pyplot.show()

####
#END OF DATAPREP
####

#define the discriminator model
def define_discriminator(image_shape):
    #weight initialization
    init = RandomNormal(stddev=0.02)
    #source image input
    in_src_image = Input(shape=image_shape)
    #target image input
    in_target_image = Input(shape =image_shape)
    #concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])
    #C64
    d = Conv2D(64, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    #C128
    d = Conv2D(128, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    #C256
    d = Conv2D(256, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(d)
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
    model = Model([in_src_image, in_target_image], patch_out)
    #compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
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
def define_generator(image_shape=(256, 256, 3)):
    #weight initialization
    init = RandomNormal(stddev=0.02)
    #image input
    in_image = Input(shape=image_shape)
    #encoder model
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)
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
    model = Model(in_image, out_image)
    return model

def define_gan(g_model, d_model, image_shape):
    #make weights in the discriminator not trainable
    d_model.trainable = False
    #define source image
    in_src = Input(shape=image_shape)
    #connect the source image to the generator input
    gen_out = g_model(in_src)
    #connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])
    #src image as input, generated image and classification output
    model = Model(in_src, [dis_out, gen_out])
    #compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=["binary_crossentropy", "mae"], optimizer=opt, loss_weights=[1,100])
    return model

#load and prepare training images
def load_real_samples(filename):
    #load the compressed arrays
    data = load(filename)
    #unpack the arrays
    X1, X2 = data["arr_0"], data["arr_1"]
    #scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return[X1, X2]

#select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    #unpack dataset
    trainA, trainB = dataset
    #choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    #retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    #generate "real" class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y

#generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    #generate fake instance
    X = g_model.predict(samples)
    #create "fake" class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

####
#END OF GAN DEFINITION
####

#generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_samples=3):
    #select a sample of input images
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    #generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    #scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
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
    #save plot to file
    filename1 = "__ganResults\\radiance1\\plot_%06d.png" % (step + 1)
    pyplot.savefig(filename1)
    pyplot.close()
    #save the generator model
    filename2 = "_models\\_radiance1\\model%06d.h5" % (step + 1)
    g_model.save(filename2)
    print(">Saved: %s and %s" % (filename1, filename2))
    
#####
#END OF Save photos incrementally
#####

#train pix2pix model
def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
    #determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    #unpack dataset
    trainA, trainB = dataset
    #calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    #calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    #manually enumerate epochs
    for i in range(n_steps):
        #select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        print(">%d, d1[%.3f] d2[%.3f] g[%.3f]" % (i+1, d_loss1, d_loss2, g_loss))
        # summarize model performance
        if (i+1) % (bat_per_epo * 2) == 0:
            summarize_performance(i, g_model, dataset)
            
####
####END OF DEFINITIONS
####

# load image data
dataset = load_real_samples("_datasets\\_radiance1\\rad_256.npz")
print("Loaded ", dataset[0].shape, dataset[1].shape)

# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)
# train model
train(d_model, g_model, gan_model, dataset)