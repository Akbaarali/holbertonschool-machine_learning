#!/usr/bin/env python3
"""Wasserstein GAN with gradient penalty."""

import tensorflow as tf
from tensorflow import keras


class WGAN_GP(keras.Model):
    """Wasserstein GAN with gradient penalty."""

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005, lambda_gp=10):
        """Initialize the WGAN_GP model."""
        super().__init__()

        self.latent_generator = latent_generator
        self.real_examples = tf.cast(real_examples, tf.float32)
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate

        self.beta_1 = .3
        self.beta_2 = .9

        self.lambda_gp = lambda_gp
        self.dims = self.real_examples.shape
        self.len_dims = len(self.dims)
        self.axis = list(range(1, self.len_dims))

        self.scal_shape = [self.batch_size]
        for _ in range(1, self.len_dims):
            self.scal_shape.append(1)

        self.generator.loss = lambda x: -tf.math.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss
        )

        self.discriminator.loss = lambda x, y: (
            tf.math.reduce_mean(x) - tf.math.reduce_mean(y)
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss
        )

    def get_fake_sample(self, size=None, training=False):
        """Generate fake samples."""
        if size is None:
            size = self.batch_size

        latent_sample = self.latent_generator(size)
        fake_sample = self.generator(latent_sample, training=training)

        return tf.cast(fake_sample, tf.float32)

    def get_real_sample(self, size=None):
        """Get random real samples."""
        if size is None:
            size = self.batch_size

        indices = tf.range(tf.shape(self.real_examples)[0])
        indices = tf.random.shuffle(indices)[:size]

        return tf.gather(self.real_examples, indices)

    def get_interpolated_sample(self, real_sample, fake_sample):
        """Create interpolated samples between real and fake samples."""
        real_sample = tf.cast(real_sample, tf.float32)
        fake_sample = tf.cast(fake_sample, tf.float32)

        shape = [tf.shape(real_sample)[0]]
        for _ in range(1, self.len_dims):
            shape.append(1)

        alpha = tf.random.uniform(shape, dtype=tf.float32)

        return alpha * real_sample + (1 - alpha) * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """Compute the gradient penalty."""
        with tf.GradientTape() as tape:
            tape.watch(interpolated_sample)
            prediction = self.discriminator(
                interpolated_sample,
                training=True
            )

        gradients = tape.gradient(prediction, interpolated_sample)
        norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=self.axis))
        gp = tf.reduce_mean((norm - 1) ** 2)

        return gp

    def train_step(self, useless_argument):
        """Perform one WGAN-GP training step."""
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_sample = self.get_real_sample()
                fake_sample = self.get_fake_sample(training=False)

                interpolated_sample = self.get_interpolated_sample(
                    real_sample,
                    fake_sample
                )

                fake_output = self.discriminator(fake_sample, training=True)
                real_output = self.discriminator(real_sample, training=True)

                discr_loss = self.discriminator.loss(
                    fake_output,
                    real_output
                )

                gp = self.gradient_penalty(interpolated_sample)
                new_discr_loss = discr_loss + self.lambda_gp * gp

            gradients = tape.gradient(
                new_discr_loss,
                self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(gradients, self.discriminator.trainable_variables)
            )

        with tf.GradientTape() as tape:
            fake_sample = self.get_fake_sample(training=True)
            fake_output = self.discriminator(fake_sample, training=False)
            gen_loss = self.generator.loss(fake_output)

        gradients = tape.gradient(
            gen_loss,
            self.generator.trainable_variables
        )
        self.generator.optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables)
        )

        return {
            "discr_loss": discr_loss,
            "gen_loss": gen_loss,
            "gp": gp
        }
