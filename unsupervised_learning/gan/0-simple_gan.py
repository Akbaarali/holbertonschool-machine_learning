#!/usr/bin/env python3
"""0-simple_gan.py - Simple GAN implementation"""
import tensorflow as tf
from tensorflow import keras
import numpy as np

class Simple_GAN(keras.Model):
    """Simple_GAN"""

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.001):
        """__init__"""
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate

        self.generator.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
        )
        self.discriminator.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
        )

    def get_fake_sample(self, size=None, training=False):
        """get_fake_sample"""
        if not size:
            size = self.batch_size
        latent_sample = self.latent_generator(size)
        return self.generator(latent_sample, training=training)

    def get_real_sample(self, size=None):
        """get_real_sample"""
        if not size:
            size = self.batch_size
        indices = tf.random.uniform(
            shape=(size,),
            minval=0,
            maxval=tf.shape(self.real_examples)[0],
            dtype=tf.int32
        )
        return tf.gather(self.real_examples, indices)

    def train_step(self, useless_argument):
        """train_step"""
        for _ in range(self.disc_iter):
            real_sample = self.get_real_sample()
            fake_sample = self.get_fake_sample(training=True)

            with tf.GradientTape() as tape:
                discr_real = self.discriminator(real_sample, training=True)
                discr_fake = self.discriminator(fake_sample, training=True)
                discr_loss = tf.keras.losses.MeanSquaredError()(discr_real, tf.ones_like(discr_real)) + \
                             tf.keras.losses.MeanSquaredError()(discr_fake, -tf.ones_like(discr_fake))

            grads = tape.gradient(discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables)
            )

        with tf.GradientTape() as tape:
            fake_sample = self.get_fake_sample(training=True)
            discr_fake = self.discriminator(fake_sample, training=False)
            gen_loss = tf.keras.losses.MeanSquaredError()(discr_fake, tf.ones_like(discr_fake))

        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}

def compare_losses(list1, list2, threshold=0.1):
    """compare_losses"""
    for i in range(len(list1)):
        if np.all(list1[i] == 0):
            continue
        diff = np.abs(np.array(list1[i]) - np.array(list2[i]))
        if np.any(diff > threshold):
            return False
    return True
