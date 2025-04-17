import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
import numpy as np

class Resnet1DBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters, type='encode'):
        super(Resnet1DBlock, self).__init__()
        if type == 'encode':
            self.conv1a = layers.Conv1D(filters, kernel_size, 2, padding="same")
            self.conv1b = layers.Conv1D(filters, kernel_size, 1, padding="same")
            self.norm1a = tfa.layers.InstanceNormalization()
            self.norm1b = tfa.layers.InstanceNormalization()
        elif type == 'decode':
            self.conv1a = layers.Conv1DTranspose(filters, kernel_size, 1, padding="same")
            self.conv1b = layers.Conv1DTranspose(filters, kernel_size, 1, padding="same")
            self.norm1a = layers.BatchNormalization()
            self.norm1b = layers.BatchNormalization()

    def call(self, input_tensor):
        x = tf.nn.relu(input_tensor)
        x = self.conv1a(x)
        x = self.norm1a(x)
        x = layers.LeakyReLU(0.4)(x)
        x = self.conv1b(x)
        x = self.norm1b(x)
        x = layers.LeakyReLU(0.4)(x)
        x += input_tensor
        return tf.nn.relu(x)

class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(1, 90001)),
            layers.Conv1D(64, 1, 2),
            Resnet1DBlock(64, 1),
            layers.Conv1D(128, 1, 2),
            Resnet1DBlock(128, 1),
            layers.Conv1D(128, 1, 2),
            Resnet1DBlock(128, 1),
            layers.Conv1D(256, 1, 2),
            Resnet1DBlock(256, 1),
            layers.Flatten(),
            layers.Dense(latent_dim + latent_dim)
        ])

        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Reshape(target_shape=(1, latent_dim)),
            Resnet1DBlock(512, 1, 'decode'),
            layers.Conv1DTranspose(512, 1, 1),
            Resnet1DBlock(256, 1, 'decode'),
            layers.Conv1DTranspose(256, 1, 1),
            Resnet1DBlock(128, 1, 'decode'),
            layers.Conv1DTranspose(128, 1, 1),
            Resnet1DBlock(64, 1, 'decode'),
            layers.Conv1DTranspose(64, 1, 1),
            layers.Conv1DTranspose(90001, 1, 1),
        ])

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(200, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        return tf.sigmoid(logits) if apply_sigmoid else logits
