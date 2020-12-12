from os import listdir
from numpy import asarray, savez_compressed, load, zeros, ones, linspace, vstack, clip, dot, sin, arccos, mean, expand_dims
from numpy.random import randn, randint
from numpy.linalg import norm
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
from PIL import Image
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
    
class DataPrep:        
        
    # load an image as an rgb numpy array
    def load_image(self, filename):
        # load image from file
        image = Image.open(filename)
        # convert to RGB, if needed
        image = image.convert("RGB")
        # convert to array
        pixels = asarray(image)
        return pixels

    # extract the face from a loaded image and resize
    def extract_face(self, model, pixels, required_size=(80, 80)):
        # detect face in the image
        faces = model.detect_faces(pixels)
        # skip cases where we could not detect a face
        if len(faces) == 0:
            return None
        # extract details of the face
        x1, y1, width, height = faces[0]["box"]
        # force detected pixel values to be positive (bug fix)
        x1, y1 = abs(x1), abs(y1)
        # convert into coordinates
        x2, y2 = x1 + width, y1 + height
        # retrieve face pixels
        face_pixels = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face_pixels)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array


    # load images and extract faces for all images in a directory
    def load_faces(self, directory, n_faces):    
        # prepare model
        model = MTCNN()
        faces = list()
        # enumerate files
        for filename in listdir(directory):
            # load the image
            pixels = self.load_image(directory + filename)
            #get face
            face = self.extract_face(model, pixels)
            if face is None:
                continue
            #store
            faces.append(face)
            print(len(faces), face.shape)
            #stop once we have enough
            if len(faces) >= n_faces:
                break
        return asarray(faces)
            
    # plot a list of loaded faces
    def plot_faces(self, faces, n):
        for i in range(n * n):
            # define subplot
            pyplot.subplot(n, n, 1 + i)
            # turn off axis
            pyplot.axis("off")
            # plot raw pixel data   
            pyplot.imshow(faces[i])
        pyplot.show()


    def dataprep(self):
        # directory that contains all images
        directory = "_datasets\\_celebA\\img_align_celeba\\"
        # load and extract all faces
        all_faces = self.load_faces(directory, 50000)
        print("Loaded: " , all_faces.shape)
        # save in compressed format
        savez_compressed("_datasets\\_celebA\\img_align_celeba.npz", all_faces)

class Train:
    #define standalone discriminator model
    def define_discriminator(self, in_shape=(80,80,3)):
        model = Sequential()
        #normal
        model.add(Conv2D(128, (5,5), padding="same", input_shape=in_shape))
        model.add(LeakyReLU(alpha=0.2))
        #downsample to 40x40
        model.add(Conv2D(128, (5,5), strides=(2,2), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        #downsample to 20x20
        model.add(Conv2D(128, (5,5), strides=(2,2), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        #downsample to 10x10
        model.add(Conv2D(128, (5,5), strides=(2,2), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        #downsample to 5x5
        model.add(Conv2D(128, (5,5), strides=(2,2), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        #classifier
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1, activation="sigmoid"))
        #compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model

    #define standalone generator model
    def define_generator(self, latent_dim):
        model = Sequential()
        #foundations for 5x5 feature maps
        n_nodes = 128 * 5 * 5
        model.add(Dense(n_nodes, input_dim=latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((5, 5, 128)))
        #upsample to 10x10
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        #upsample 20x20
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        #upsample 40x40
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        #upsample 80x80
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        #output layer 80x80x3
        model.add(Conv2D(3, (5,5), activation="tanh", padding="same"))
        return model
        
    #define the combined generator and discriminator model, for updating generator
    def define_gan(self, g_model, d_model):
        #make weights in the discriminator not trainable
        d_model.trainable = False
        #connect them
        model = Sequential()
        #add generator
        model.add(g_model)
        #add discriminator
        model.add(d_model)
        #compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss="binary_crossentropy", optimizer=opt)
        return model
    
    #load and prepare training images
    def load_real_samples(self):
        #load real the face dataset
        data = load("_datasets\\_celebA\\img_align_celeba.npz")
        X = data["arr_0"]
        #convert from unsigned ints to floats
        X = X.astype("float32")
        #scale from [0, 255] to [-1, 1]
        X = (X - 127.5) / 127.5
        return X
    
    #select real samples
    def generate_real_samples(self, dataset, n_samples):
        #choose random instances
        ix = randint(0, dataset.shape[0], n_samples)
        #retrieve selected images
        X = dataset[ix]
        #generate "real" class labels (1)
        y = ones((n_samples, 1))
        return X, y

    def generate_latent_points(self, latent_dim, n_samples):
        #generate points in the latent space
        x_input = randn(latent_dim * n_samples)
        #reshape into a batch of inputs for the network
        x_input = x_input.reshape(n_samples, latent_dim)
        return x_input
    
    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(self, g_model, latent_dim, n_samples):
        #generate points in latent space
        x_input = self.generate_latent_points(latent_dim, n_samples)
        #predict outputs
        X = g_model.predict(x_input)
        #create "fake" class labels (0)
        y = zeros((n_samples, 1))
        return X, y
    
    # create and save a plot of generated images
    def save_plot(self, examples, epoch, n=4):
        # scale from [-1,1] to [0,1]
        examples = (examples + 1) / 2.0
        # plot images
        for i in range(n * n):
            # define subplot
            pyplot.subplot(n, n, 1 + i)
            # turn off axis
            pyplot.axis("off")
            # plot raw pixel data
            pyplot.imshow(examples[i])
        # save plot to file
        filename = "__ganResults\\CelebAGAN\\generated_plot_e%03d.png" % (epoch+1)
        pyplot.savefig(filename)
        pyplot.close()
        
    # evaluate the discriminator, plot generated images, save generator model
    def summarize_performance(self, epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
        # prepare real samples
        X_real, y_real = self.generate_real_samples(dataset, n_samples)
        # evaluate discriminator on real examples
        _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
        # prepare fake examples
        x_fake, y_fake = self.generate_fake_samples(g_model, latent_dim, n_samples)
        # evaluate discriminator on fake examples
        _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
        # summarize discriminator performance
        print(">Accuracy real: %.0f%%, fake: %.0f%%" % (acc_real*100, acc_fake*100))
        # save plot
        self.save_plot(x_fake, epoch)
        # save the generator model tile file
        filename = "_models\\_CelebAGAN\\generator_model_%03d.h5" % (epoch+1)
        if (epoch + 1 ) % 4 == 0:
            g_model.save(filename)

    
    #train generator and discriminator
    def train_gan(self, g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=100):
        bat_per_epo = int(dataset.shape[0] / n_batch)
        half_batch = int(n_batch / 2)
        #manually enumerate epochs
        for i in range(n_epochs):
            #enumerate batches over the training set
            for j in range(bat_per_epo):
                #get randomly selected "real" samples
                X_real, y_real = self.generate_real_samples(dataset, half_batch)
                #update discriminator model weights
                d_loss1, _ = d_model.train_on_batch(X_real, y_real)
                #generate "fake" examples
                X_fake, y_fake = self.generate_fake_samples(g_model, latent_dim, half_batch)
                #update discriminator model weights
                d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
                #prepare points in latent space as input for the generator
                X_gan = self.generate_latent_points(latent_dim, n_batch)
                #create inverted labels for the fake samples
                y_gan = ones((n_batch, 1))
                #update generator via the discriminator's error
                g_loss = gan_model.train_on_batch(X_gan, y_gan)
                print(">Epoch: %d, Batch: %d/%d, d_real_loss: %.3f, d_fake_loss: %.3f, gan_loss = %.3f" % 
                                                        (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
            # evaluate the model performance, sometimes save model
            self.summarize_performance(i, g_model, d_model, dataset, latent_dim)
    
    def train(self):
        # size of the latent space
        latent_dim = 100
        # create the discriminator
        d_model = self.define_discriminator()
        # create the generator
        g_model = self.define_generator(latent_dim)
        # create the gan
        gan_model = self.define_gan(g_model, d_model)
        # load image data
        dataset = self.load_real_samples()
        # train model
        self.train_gan(g_model, d_model, gan_model, dataset, latent_dim)

class FaceGenerator:
    def __init__(self, model_name, n_examples):
        self.model_name = model_name
        self.n_examples = n_examples
        
    # generate points in latent space as input for the generator
    def generate_latent_points(self, latent_dim, n_samples):
        # generate points in the latent space
        x_input = randn(latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        z_input = x_input.reshape(n_samples, latent_dim)
        return z_input
    
    # create a plot of generated images
    def plot_generated(self, examples, n):
        # plot images
        for i in range(n * n):
            # define subplot
            pyplot.subplot(n, n, 1 + i)
            # turn off axis
            pyplot.axis("off")
            # plot raw pixel data
            pyplot.imshow(examples[i, :, :])
        pyplot.show()

    def generate_faces(self):
        # load model
        model = load_model("_models\\_CelebAGAN\\generator_model_" + str(self.model_name) +".h5")
        # generate latent poşnts
        latent_points = self.generate_latent_points(100, 25)
        # generate images
        X = model.predict(latent_points)
        # scale from [-1,1] to [0,1]
        X = (X + 1) / 2.0
        # plot the result
        self.plot_generated(X, self.n_examples)

class LatentSpaceInterpolater:
    def __init__(self, model_name, number_of_rows):
        self.number_of_rows = number_of_rows 
        self.model_name = model_name
        
    #generate points in latent space as input for the generator
    def generate_latent_points(self, latent_dim, n_samples):
        #generate points in the latent space
        x_input = randn(latent_dim * n_samples)
        #reshape into a batch of inputs for the network
        z_input = x_input.reshape(n_samples, latent_dim)
        return z_input
    
    #uniform interpolation between two points in latent space
    def interpolate_points(self, p1, p2, n_steps=10):
        # interpolate ratios between the points
        ratios = linspace(0, 1, num=n_steps)
        # linear interpolate vectors
        vectors = list()
        for ratio in ratios:
            v = (1.0 - ratio) * p1 + ratio * p2
            vectors.append(v)
        return asarray(vectors)
    
    #create a plot of generated images
    def plot_generated(self, examples, n):
        #plot images
        for i in range(n * 10):
            #define subplot
            pyplot.subplot(n, 10, 1 + i)
            #turn off axis
            pyplot.axis("off")
            #plot raw pixel data
            pyplot.imshow(examples[i, :, :])
        pyplot.show()
    
    #linear interpolate latent space to see multiple faces (n)    
    def lerp_latent_space(self):
        # load model
        model = load_model("_models\\_CelebAGAN\\generator_model_" + self.model_name +".h5")
        pts = self.generate_latent_points(100, self.number_of_rows * 2)
        results = None
        #interpolate pairs
        for i in range(0, self.number_of_rows * 2, 2):
            #interpolate points in latent space
            interpolated = self.interpolate_points(pts[i], pts[i + 1])
            #generate images
            X = model.predict(interpolated)
            #scale from [-1, 1] to [0, 1]
            X = (X + 1) / 2.0
            if results is None:
                results = X
            else:
                results = vstack((results, X))
        #plot the result
        self.plot_generated(results, self.number_of_rows)
    
    #spherical linear interpolation (slerp)
    def slerp(self, val, low, high):
        omega =  arccos(clip(dot(low/norm(low), high/norm(high)), -1, 1))
        so = sin(omega)
        if so == 0:
            # L'Hopital's rule/LERP
            return (1.0-val) * low + val * high
        return sin((1.0-val)*omega) / so * low + sin(val*omega) / so * high
    
    def slerp_points(self, p1, p2, n_steps=10):
        # interpolate ratios between the points
        ratios = linspace(0, 1, num=n_steps)
        # spherical interpolate vectors
        vectors = list()
        for ratio in ratios:
            v = self.slerp(ratio, p1, p2)
            vectors.append(v)
        return asarray(vectors)
    
    def slerp_latent_space(self):            
        # load model
        model = load_model("_models\\_CelebAGAN\\generator_model_" + self.model_name +".h5")
        pts = self.generate_latent_points(100, self.number_of_rows * 2)
        results = None
        #interpolate pairs
        for i in range(0, self.number_of_rows * 2, 2):
            #interpolate points in latent space
            interpolated = self.slerp_points(pts[i], pts[i + 1])
            #generate images
            X = model.predict(interpolated)
            #scale from [-1, 1] to [0, 1]
            X = (X + 1) / 2
            if results is None:
                results = X
            else:
                results = vstack((results, X))
        #plot the result
        self.plot_generated(results, self.number_of_rows)

class LatentSpaceExplorer:
    def __init__(self,  model_name, number_of_rows):
            self.model_name = model_name
            self.number_of_rows = number_of_rows 
            
    #generate points in latent space as input for the generator
    def generate_latent_points(self, latent_dim, n_samples):
        #generate points in the latent space
        x_input = randn(latent_dim * n_samples)
        #reshape into a batch of inputs for the network
        z_input = x_input.reshape(n_samples, latent_dim)
        return z_input
    
    #create a plot of generated images
    def save_plot_generated(self, examples, n):
        #plot images
        for i in range(n * 10):
            #define subplot
            pyplot.subplot(n, 10, 1 + i)
            #turn off axis
            pyplot.axis("off")
            #plot raw pixel data
            pyplot.imshow(examples[i, :, :])
        pyplot.savefig("__ganResults\\CelebAGAN\\LatentSpaceExploration\\generated_faces.png")
        pyplot.close()
        
        #create a plot of generated images
    def plot_generated(self, examples, rows, cols):
        #plot images
        for i in range(rows * cols):
            #define subplot
            pyplot.subplot(rows, cols, 1 + i)
            #turn off axis
            pyplot.axis("off")
            #plot raw pixel data
            pyplot.imshow(examples[i, :, :])
        pyplot.show()

    def save_plot_pts(self):
        # load model
        model = load_model("_models\\_CelebAGAN\\generator_model_" + self.model_name +".h5")
        latent_points = self.generate_latent_points(100, self.number_of_rows * 10)
        # save points
        savez_compressed("__ganResults\\CelebAGAN\\LatentSpaceExploration\\latent_points.npz", latent_points)
        # generate images
        X = model.predict(latent_points)
        # scale from [-1,1] to [0,1]
        X = (X + 1) / 2.0
        self.save_plot_generated(X, self.number_of_rows)
    
    def average_points(self, points, ix):
        #retrieve required points
        vectors = points[ix]
        # average the vectors
        avg_vector = mean(vectors, axis=0)
        # combine original and avg vectors
        all_vectors = vstack((vectors, avg_vector))
        return all_vectors
    
    def latent_space_arithmetic(self):
        # load model
        model = load_model("_models\\_CelebAGAN\\generator_model_" + self.model_name +".h5")
        # retrieve specific points
        smiling_woman_ix = [0, 3, 38]
        neutral_woman_ix = [4, 19, 39]
        neutral_man_ix = [10, 15, 33]
        # load the saved latent points
        data = load("__ganResults\\CelebAGAN\\LatentSpaceExploration\\latent_points.npz")
        points = data["arr_0"]
        #average vectors
        smiling_woman = self.average_points(points, smiling_woman_ix)
        neutral_woman = self.average_points(points, neutral_woman_ix)
        neutral_man = self.average_points(points, neutral_man_ix)
        # combine all vectors
        all_vectors = vstack((smiling_woman, neutral_woman, neutral_man))
        # generate images
        images = model.predict(all_vectors)
        # scale pixel values
        images = (images + 1) / 2.0
        self.plot_generated(images, 3, 4)
        # smiling woman - neutral woman + neutral man = smiling man
        result_vector = smiling_woman[-1] - neutral_woman[-1] + neutral_man[-1]
        # generate image
        result_vector = expand_dims(result_vector, 0)
        result_image = model.predict(result_vector)
        # scale pixel values
        result_image = (result_image + 1) / 2.0
        pyplot.imshow(result_image[0])
        pyplot.show()

        
###############
#####PREP DATA
###############

""" dataprepper = DataPrep()
dataprepper.dataprep() """

###############
######TRAIN
###############

""" trainer = Train()
trainer.train() """

###############
######GENERATE FACES
###############

""" face_generator = FaceGenerator("052" , 5)
face_generator.generate_faces() """

###############
######INTERPOLATE LATENT SPACE
###############

""" interpolator = LatentSpaceInterpolater("052", 5)
interpolator.slerp_latent_space() """

###############
######EXPLORE LATENT SPACE
###############

""" explorer = LatentSpaceExplorer("052", 5)
#explorer.save_plot_pts()
explorer.latent_space_arithmetic()  """