from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten,  Lambda
from keras.optimizers import SGD, Adam
from keras import backend as K

from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math

from keras.layers.advanced_activations import LeakyReLU
from skimage.transform import resize

def resize_set(x, w, h, **kw):
    x_out = np.empty((x.shape[0], 1, w, h))
    for i in range(len(x)):
        x_out[i, 0] = resize(x[i, 0], (w, h), **kw)
    return x_out.astype(np.float32)

leaky_relu = LeakyReLU(alpha=0.3)

def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(leaky_relu)
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(leaky_relu)
    model.add(Reshape((128, 7, 7), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(leaky_relu)
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(1, 5, 5, border_mode='same'))
    model.add(Activation('sigmoid'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(Convolution2D(
                        64, 5, 5,
                        border_mode='same',
                        input_shape=(1, 28, 28)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 5, 5))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def sample_std_objectness(v):
    eps = 1e-8
    marginal = v.mean(axis=0)
    score = (v).std(axis=1)
    return score 

def sample_objectness_v2(v):
    return v[:, 0]

def sample_objectness(v):
    eps = 1e-8
    #marginal = v.mean(axis=0)
    #score = (v * K.log(v / (marginal + eps)))
    score = v * K.log(v)
    score = score.sum(axis=1)
    return score 
   
def objectness(v):
    eps = 1e-8
    marginal = v.mean(axis=0)
    score = (v * K.log(v / (marginal + eps)))
    score = score.sum(axis=1).mean()
    return score 

def output_shape_objectness(input_shape):
    return (input_shape[0], 1)

def output_shape_class_get(input_shape):
    return (input_shape[0], 1)

def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    #model.add(Lambda(objectness, output_shape=output_shape_objectness))
    #model.add(Lambda(class_get, output_shape=output_shape_class_get))
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[2:]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[0, :, :]
    return image



def train(BATCH_SIZE):
    from keras.models import model_from_json 
    from datakit import mnist
    """
    data = mnist.load(which='train')
    X_train = data['train']['X']
    y_train = data['train']['y']
    X_train = X_train.astype(np.float32) / 255.
    iprint(X_train.min(), X_train.max())
    X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])
    X_train = X_train[(y_train==6)|(y_train==9)] 
    """
    X, y = np.load('/home/mcherti/work/data/fonts/ds_all_32.npy')
    X = np.array(X.tolist())
    y = np.array(y.tolist())
    X = X.reshape((X.shape[0], 1, 32, 32))
    X = X.astype(np.float32)
    X = resize_set(X, 28, 28)
    indices = np.arange(len(X))
    np.random.shuffle(X)
    X = X[indices]
    X_train = X
    print(X_train.shape)
    #discriminator = discriminator_model()
    js = open('../../feature_generation/tools/models/mnist/m2/model.json').read()
    js = js.replace('softmax', 'linear')
    discriminator = model_from_json(js)
    discriminator.load_weights('../../feature_generation/tools/models/mnist/m2/model.h5')
    generator = generator_model()

    discriminator_on_generator = \
        generator_containing_discriminator(generator, discriminator)
    
    #d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    d_optim = Adam(lr=0.0002, beta_1=0.5)
    #g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = Adam(lr=0.0002, beta_1=0.5)

    #d_optim = adam(lr=0)
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    def loss(y_true, y_pred):
        return (-sample_objectness_v2(y_pred)).mean()
    discriminator_on_generator.compile(
        loss=loss, optimizer=g_optim)
    
    discriminator.trainable = False
    def loss(y_true, y_pred):
        y_true = y_true[:, 0]
        return (  y_true    *    (-sample_objectness_v2(y_pred))       ).mean()
    discriminator.compile(loss=loss, optimizer=d_optim)
    noise = np.zeros((BATCH_SIZE, 100))
    for epoch in range(1000000):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = generator.predict(noise, verbose=0)
            if index % 20 == 0:
                image = combine_images(generated_images)
                image = image*255.
                Image.fromarray(image.astype(np.uint8)).save(
                    'modif/{:04d}{:05d}.png'.format(epoch, index))
            X = np.concatenate((image_batch, generated_images))
            y = np.ones((X.shape[0], 10))
            y[0:BATCH_SIZE] = 1
            y[BATCH_SIZE:] = -1
            d_loss = discriminator.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(
                noise, [1] * BATCH_SIZE)
            discriminator.trainable = False
            print("batch %d g_loss : %f" % (index, g_loss))
            #if index % 10 == 9:
            #    generator.save_weights('generator', True)
            #    discriminator.save_weights('discriminator', True)

def generate(BATCH_SIZE, nice=False):
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('generator')
    if nice:
        discriminator = discriminator_model()
        discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
        discriminator.load_weights('discriminator')
        noise = np.zeros((BATCH_SIZE*20, 100))
        for i in range(BATCH_SIZE*20):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        d_pret = discriminator.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE, 1) +
                               (generated_images.shape[2:]), dtype=np.float32)
        for i in range(int(BATCH_SIZE)):
            idx = int(pre_with_index[i][1])
            nice_images[i, 0, :, :] = generated_images[idx, 0, :, :]
        image = combine_images(nice_images)
    else:
        noise = np.zeros((BATCH_SIZE, 100))
        for i in range(BATCH_SIZE):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
