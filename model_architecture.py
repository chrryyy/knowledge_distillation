#Initial VAE adapted from https://keras.io/examples/generative/vae/

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import ops
from keras import layers
from keras import models
from keras.models import Model

#Sampling layer to sample z from the gaussian output by the encoder
class Sampling(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    def get_config(self):
        # Save encoder and decoder configuration
        config = super().get_config()
        config.update({
            "encoder": self.encoder.get_config(),
            "decoder": self.decoder.get_config(),
        })
        return config

    @classmethod
    def from_config(cls, config):
        encoder = Model.from_config(config.pop("encoder"))
        decoder = Model.from_config(config.pop("decoder"))
        return cls(encoder=encoder, decoder=decoder, **config)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = ops.mean(
                ops.sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
    }

class BetaScheduler(keras.callbacks.Callback):
    def __init__(self, beta_var, max_beta, schedule_epochs):
        super().__init__()
        self.beta_var = beta_var
        self.max_beta = max_beta
        self.schedule_epochs = schedule_epochs

    def on_epoch_end(self, epoch, logs=None):
        new_beta = min(self.max_beta, (epoch + 1) / self.schedule_epochs * self.max_beta)
        self.beta_var.assign(new_beta)


class BetaVAE(keras.Model):
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    def get_config(self):
        # Save encoder and decoder configuration, along with beta value
        config = super().get_config()
        config.update({
            "encoder_config": self.encoder.get_config(),
            "decoder_config": self.decoder.get_config(),
            "beta": self.beta,
        })
        return config

    @classmethod
    def from_config(cls, config):
        encoder = Model.from_config(config.pop("encoder_config"))
        decoder = Model.from_config(config.pop("decoder_config"))
        return cls(encoder=encoder, decoder=decoder, **config)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + self.beta * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


class TC_VAE(keras.Model):
    def __init__(self, encoder, decoder, beta=1.0, beta_tc=10.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.beta_tc = beta_tc  # TC weight
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.tc_loss_tracker = keras.metrics.Mean(name="tc_loss")

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    def get_config(self):
        config = super().get_config()
        config.update({
            "encoder": self.encoder.get_config(),
            "decoder": self.decoder.get_config(),
            "beta": self.beta,
            "beta_tc": self.beta_tc,
        })
        return config

    @classmethod
    def from_config(cls, config):
        from keras.models import Model
        encoder = Model.from_config(config.pop("encoder"))
        decoder = Model.from_config(config.pop("decoder"))
        return cls(encoder=encoder, decoder=decoder, **config)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.tc_loss_tracker,
        ]

    def compute_tc_loss(self, z):
        # Estimate joint q(z) and marginals q(z_i)
        q_z = tf.reduce_mean(tf.exp(-0.5 * tf.reduce_sum(tf.square(z), axis=1)))  # Joint posterior
        q_z_marginals = tf.reduce_prod(tf.reduce_mean(tf.exp(-0.5 * tf.square(z)), axis=0))  # Product of marginals
        return tf.math.log(q_z / q_z_marginals)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # Compute reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
            )

            # Compute KL loss
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )

            # Compute TC loss
            tc_loss = self.compute_tc_loss(z)

            # Combine losses
            total_loss = reconstruction_loss + self.beta * kl_loss + self.beta_tc * tc_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.tc_loss_tracker.update_state(tc_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "tc_loss": self.tc_loss_tracker.result(),
        }

#Builds off of the basic VAE and adds a distillation loss term instead
class DistillationVAE(VAE):
    def __init__(self, encoder, decoder, teacher_model=None, alpha=1.0, **kwargs):
        super().__init__(encoder, decoder, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.distillation_loss_tracker = keras.metrics.Mean(name="distillation_loss")

    def get_config(self):
        config = super().get_config()
        config.update({
            "encoder_config": self.encoder.get_config(),
            "decoder_config": self.decoder.get_config(),
            "alpha": self.alpha,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Recreates the DistillationVAE from its configuration.
        """
        # Extract encoder and decoder configs
        encoder_config = config.pop("encoder_config")
        decoder_config = config.pop("decoder_config")
        alpha = config.pop("alpha", 1.0)

        # Rebuild encoder and decoder
        encoder = Model.from_config(encoder_config)
        decoder = Model.from_config(decoder_config)

        # Pass only remaining configs to the constructor
        return cls(encoder=encoder, decoder=decoder, alpha=alpha)


    @property
    def metrics(self):
        return super().metrics + [self.distillation_loss_tracker]