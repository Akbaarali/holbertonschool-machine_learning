#!/usr/bin/env python3
"""WGAN-GP with support for loading pre-trained weights."""

WGAN_GP_Base = __import__('2-wgan_gp').WGAN_GP


class WGAN_GP(WGAN_GP_Base):
    """Wasserstein GAN with gradient penalty and weight loading."""

    def replace_weights(self, gen_h5, disc_h5):
        """Replace generator and discriminator weights from H5 files."""
        self.generator.load_weights(gen_h5)
        self.discriminator.load_weights(disc_h5)

    def replace_weight(self, gen_h5, disc_h5):
        """Alias for replace_weights."""
        self.replace_weights(gen_h5, disc_h5)
