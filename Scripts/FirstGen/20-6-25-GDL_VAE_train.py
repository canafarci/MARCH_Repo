import os
import tensorflow as tf

from MLobjects.AE import Autoencoder
import numpy as np

# run params
SECTION = 'vae'
RUN_ID = '0002'
DATA_NAME = 'digits'
RUN_FOLDER = 'D:/ders/MASTER/_Thesis/VirtualEnv/_models/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

MODE = 'build'  # 'load' #

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = np.array(x_train).reshape(-1, 28, 28, 1)

AE = Autoencoder(
    input_dim=(28, 28, 1),
    encoder_conv_filters=[32, 64, 64, 64],
    encoder_conv_kernel_size=[3, 3, 3, 3],
    encoder_conv_strides=[1, 2, 2, 1],
    decoder_conv_t_filters=[64, 64, 32, 1],
    decoder_conv_t_kernel_size=[3, 3, 3, 3],
    decoder_conv_t_strides=[1, 2, 2, 1], 
    z_dim=2
    )

if MODE == 'build':
    AE.save(RUN_FOLDER)
else:
    AE.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

LEARNING_RATE = 0.0005
R_LOSS_FACTOR = 1000


AE.compile(LEARNING_RATE, R_LOSS_FACTOR)


BATCH_SIZE = 32
INITIAL_EPOCH = 0
PRINT_EVERY_N_BATCHES = 100

AE.train(
    x_train, batch_size=BATCH_SIZE, epochs=200, run_folder=RUN_FOLDER, 
    print_every_n_batches=PRINT_EVERY_N_BATCHES, initial_epoch=INITIAL_EPOCH
    )
