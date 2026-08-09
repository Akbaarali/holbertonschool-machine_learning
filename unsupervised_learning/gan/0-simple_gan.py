def train_step(self, useless_argument):
    """Perform one training step for the GAN."""

    for _ in range(self.disc_iter):
        with tf.GradientTape() as tape:
            real_sample = self.get_real_sample()
            fake_sample = self.get_fake_sample()

            real_output = self.discriminator(
                real_sample,
                training=True
            )
            fake_output = self.discriminator(
                fake_sample,
                training=True
            )

            discr_loss = self.discriminator.loss(
                real_output,
                fake_output
            )

        gradients = tape.gradient(
            discr_loss,
            self.discriminator.trainable_variables
        )

        self.discriminator.optimizer.apply_gradients(
            zip(
                gradients,
                self.discriminator.trainable_variables
            )
        )

    with tf.GradientTape() as tape:
        fake_sample = self.get_fake_sample(training=True)

        fake_output = self.discriminator(
            fake_sample,
            training=False
        )

        gen_loss = self.generator.loss(fake_output)

    gradients = tape.gradient(
        gen_loss,
        self.generator.trainable_variables
    )

    self.generator.optimizer.apply_gradients(
        zip(
            gradients,
            self.generator.trainable_variables
        )
    )

    return {
        "discr_loss": discr_loss,
        "gen_loss": gen_loss
    }
