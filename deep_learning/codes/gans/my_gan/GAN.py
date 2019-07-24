#adapted from Generative Deep Learning by David Foster

from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.merge import _Merge

from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from keras.initializers import RandomNormal

import os
import numpy as np
import matplotlib.pyplot as plt

class GAN():
    def __init__(self, input_dim):

        self.input_dim = input_dim
        self.z_dim = 100
        self.weight_init = RandomNormal(mean=0., stddev=0.02)
        self.d_losses = []
        self.g_losses = []

        self.epoch = 0

        self._build_discriminator()
        self._build_generator()

        self._build_adversarial()

    def _build_discriminator(self):

        discriminator_input = Input(shape=self.input_dim, name='discriminator_input')
        conv1 = Conv2D(64, (5, 5), activation='relu', strides=(2, 2), padding = 'same', kernel_initializer = self.weight_init)(discriminator_input)
        conv1 = Dropout(rate = 0.4)(conv1)
        conv2 = Conv2D(64, (5, 5), activation='relu', strides=(2, 2), padding = 'same', kernel_initializer = self.weight_init)(conv1)
        conv2 = Dropout(rate = 0.4)(conv2)
        conv3 = Conv2D(128, (5, 5), activation='relu', strides=(2, 2), padding = 'same', kernel_initializer = self.weight_init)(conv2)
        conv3 = Dropout(rate = 0.4)(conv3)
        conv4 = Conv2D(128, (5, 5), activation='relu', strides=(1, 1), padding = 'same', kernel_initializer = self.weight_init)(conv3)
        conv4 = Dropout(rate = 0.4)(conv4)
        conv4 = Flatten()(conv4)

        discriminator_output = Dense(1, activation='sigmoid',  kernel_initializer = self.weight_init)(conv4)
        self.discriminator = Model(discriminator_input, discriminator_output)

        opt = RMSprop(lr=0.0008)
        self.discriminator.compile(
        optimizer=opt
        , loss = 'binary_crossentropy'
        ,  metrics = ['accuracy']
        )

        self.discriminator.summary()

    def _build_generator(self):

        generator_input = Input(shape=(self.z_dim,), name='generator_input')

        dense1 = Dense(3136, activation='relu')(generator_input)
        dense1 = BatchNormalization(momentum = 0.9)(dense1)
        dense1 = Activation('relu')(dense1)

        dense1 = Reshape((7, 7, 64))(dense1)
        dense1 = Dropout(rate = 0.4)(dense1)

        up1 = UpSampling2D()(dense1)
        conv1 = Conv2D(128, (5, 5), padding='same', kernel_initializer = self.weight_init)(up1)
        conv1 = BatchNormalization(momentum = 0.9)(conv1)
        conv1 = Activation('relu')(conv1)

        up2 = UpSampling2D()(conv1)
        conv2 = Conv2D(64, (5, 5),  padding='same', kernel_initializer = self.weight_init)(up2)
        conv2 = BatchNormalization(momentum = 0.9)(conv2)
        conv2 = Activation('relu')(conv2)

        conv3 = Conv2D(64, (5, 5),  padding='same', kernel_initializer = self.weight_init)(conv2)
        conv3 = BatchNormalization(momentum = 0.9)(conv3)
        conv3 = Activation('relu')(conv3)

        conv4 = Conv2D(1, (5, 5),  padding='same')(conv3)
        conv4 = Activation('tanh')(conv4)

        generator_output = conv4

        self.generator = Model(generator_input, generator_output)

        self.generator.summary()


    def _build_adversarial(self):

        self.discriminator.trainable = False

        model_input = Input(shape=(self.z_dim,), name='model_input')
        model_output = self.discriminator(self.generator(model_input))
        self.model = Model(model_input, model_output)

        opt = RMSprop(lr=0.0004)
        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        self.discriminator.trainable = False

    def train_discriminator(self, x_train, batch_size):

        valid = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))

        idx = np.random.randint(0, x_train.shape[0], batch_size)
        true_imgs = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        d_loss_real, d_acc_real =   self.discriminator.train_on_batch(true_imgs, valid)
        d_loss_fake, d_acc_fake =   self.discriminator.train_on_batch(gen_imgs, fake)
        d_loss =  0.5 * (d_loss_real + d_loss_fake)
        d_acc = 0.5 * (d_acc_real + d_acc_fake)

        return [d_loss, d_loss_real, d_loss_fake, d_acc, d_acc_real, d_acc_fake]

    def train_generator(self, batch_size):
        valid = np.ones((batch_size,1))
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        return self.model.train_on_batch(noise, valid)

    def train(self, x_train, batch_size, epochs, run_folder
    , print_every_n_batches = 50):

        for epoch in range(self.epoch, self.epoch + epochs):

            d = self.train_discriminator(x_train, batch_size)
            g = self.train_generator(batch_size)

            print ("%d [D loss: (%.3f)(R %.3f, F %.3f)] [D acc: (%.3f)(%.3f, %.3f)] [G loss: %.3f] [G acc: %.3f]" % (epoch, d[0], d[1], d[2], d[3], d[4], d[5], g[0], g[1]))

            self.d_losses.append(d)
            self.g_losses.append(g)

            if epoch % print_every_n_batches == 0:
                self.sample_images(run_folder)
                self.save_model(run_folder)

            self.epoch += 1

    def sample_images(self, run_folder):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        fig, axs = plt.subplots(r, c, figsize=(15,15))
        cnt = 0

        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(np.squeeze(gen_imgs[cnt, :,:,:]), cmap = 'gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(run_folder, "images/sample_%d.png" % self.epoch))
        plt.close()

    def save_model(self, run_folder):
        self.model.save(os.path.join(run_folder, 'models/model.h5'))
        self.discriminator.save(os.path.join(run_folder, 'models/discriminator.h5'))
        self.generator.save(os.path.join(run_folder, 'models/generator.h5'))
