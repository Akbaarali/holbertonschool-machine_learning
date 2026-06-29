#!/usr/bin/env python3
"""1-wgan_clip.py - Wasserstein GAN with weight clipping"""
import tensorflow as tf
from tensorflow import keras
import numpy as np

class WGAN_clip(keras.Model):
    """WGAN_clip"""

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2, learning_rate=.005):
        """__init__"""
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate
        self.beta_1 = 0.5
        self.beta_2 = 0.9

        # generator loss: negative mean of discriminator output on fake samples
        self.generator.loss = lambda fake_pred: -tf.math.reduce_mean(fake_pred)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.generator.compile(optimizer=self.generator.optimizer, loss=self.generator.loss)

        # discriminator loss: mean(fake) - mean(real)
        self.discriminator.loss = lambda real_pred, fake_pred: tf.math.reduce_mean(fake_pred) - tf.math.reduce_mean(real_pred)
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.discriminator.compile(optimizer=self.discriminator.optimizer, loss=self.discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """get_fake_sample"""
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """get_real_sample"""
        if not size:
            size = self.batch_size
        indices = tf.range(tf.shape(self.real_examples)[0])
        shuffled = tf.random.shuffle(indices)[:size]
        return tf.gather(self.real_examples, shuffled)

    def train_step(self, useless_argument):
        """train_step"""
        for _ in range(self.disc_iter):
            real_sample = self.get_real_sample()
            fake_sample = self.get_fake_sample(training=True)

            with tf.GradientTape() as tape:
                discr_real = self.discriminator(real_sample, training=True)
                discr_fake = self.discriminator(fake_sample, training=True)
                discr_loss = self.discriminator.loss(discr_real, discr_fake)

            grads = tape.gradient(discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables)
            )

            # Clip discriminator weights between -1 and 1
            for var in self.discriminator.trainable_variables:
                var.assign(tf.clip_by_value(var, -1.0, 1.0))

        fake_sample = self.get_fake_sample(training=True)
        with tf.GradientTape() as tape:
            discr_fake = self.discriminator(fake_sample, training=False)
            gen_loss = self.generator.loss(discr_fake)

        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
