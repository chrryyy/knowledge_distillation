from model_architecture import BetaVAE, VAE, TC_VAE, Sampling, BetaScheduler
from keras import layers
import keras
import os
import numpy as np
import tensorflow as tf

np.random.seed(123)
replace_baseline_model = False
replace_teacher_model = False
replace_student_model = False
replace_tc_model = False

#Encoder creation function
def create_baseline_encoder(model_name, latent_dim = 5, input_shape=(28, 28, 1)):
    encoder_inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name=model_name)

def create_teacher_encoder(latent_dim=5, input_shape=(28, 28, 1)):
    encoder_inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="teacher_encoder")

#Decoder creation function, decoders are the same across all models
def create_decoder(latent_dim = 5, num_filters=64, output_shape=(28, 28, 1)):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * num_filters, activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, num_filters))(x)
    x = layers.Conv2DTranspose(num_filters, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(output_shape[-1], 3, activation="sigmoid", padding="same")(x)
    return keras.Model(latent_inputs, decoder_outputs, name=f"decoder_{num_filters}_filters")

#Load MNIST data
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
digits = np.concatenate([x_train, x_test], axis=0)
digits = np.expand_dims(digits, -1).astype("float32") / 255

if replace_baseline_model:
    if os.path.exists('models/vae_baseline.keras'):
        os.remove('models/vae_baseline.keras')
    #Basic baseline VAE: Fewer filters, simpler architecture
    encoder_baseline = create_baseline_encoder("baseline_vae")
    decoder_baseline = create_decoder(num_filters=32)
    vae_baseline = VAE(encoder_baseline, decoder_baseline)
    vae_baseline.compile(optimizer=keras.optimizers.Adam())

    print("Baseline VAE Encoder Summary:")
    encoder_baseline.summary()

    #Train and save baseline VAE
    vae_baseline.fit(digits, epochs=20, batch_size=128)
    vae_baseline.save('models/vae_baseline.keras')

if replace_teacher_model:
    if os.path.exists('models/bvae_teacher.keras'):
        os.remove('models/bvae_teacher.keras')
    #Teacher VAE: More filters, more complex architecture
    encoder_teacher = create_teacher_encoder()
    decoder_teacher = create_decoder(num_filters=32)
    beta_var = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    vae_teacher = BetaVAE(encoder_teacher, decoder_teacher, beta=beta_var)
    vae_teacher.compile(optimizer=keras.optimizers.Adam())

    #Gradually increase beta from 0 to 4 over 10 epochs
    callbacks = [BetaScheduler(beta_var, max_beta=4.0, schedule_epochs=10)]
    vae_teacher.fit(digits, epochs=20, batch_size=128, callbacks=callbacks)

    vae_teacher.save('models/bvae_teacher.keras')

if replace_tc_model:
    if os.path.exists('models/tcvae_teacher.keras'):
        os.remove('models/tcvae_teacher.keras')
    #Teacher VAE: More filters, more complex architecture
    encoder_teacher = create_teacher_encoder()
    decoder_teacher = create_decoder(num_filters=32)
    vae_teacher = TC_VAE(encoder_teacher, decoder_teacher, beta=1.0, beta_tc=0.0)
    vae_teacher.compile(optimizer=keras.optimizers.Adam())

    #Add BetaScheduler
    beta_tc_var = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    callbacks = [BetaScheduler(beta_tc_var, max_beta=10.0, schedule_epochs=10)]

    #Train
    vae_teacher.fit(digits, epochs=20, batch_size=128, callbacks=callbacks)
    vae_teacher.save('models/tcvae_teacher.keras')

