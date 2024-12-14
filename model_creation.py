from model_architecture import BetaVAE, VAE, TC_VAE, DistillationVAE, Sampling, BetaScheduler
from keras import layers
import keras
import os
import numpy as np
import tensorflow as tf

np.random.seed(123)
replace_baseline_model = False

#Standard big model
replace_big_model = False

#Enhanced big model with better disentanglement
replace_tc_model = False

#Two student models
replace_student_model = False
replace_student_tc_model = False

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



import sys

#Output Logging
output_file = "output_logs/creation.txt"

with open(output_file, "w") as f:
    sys.stdout = f
    try:
        if replace_baseline_model:
            if os.path.exists('models/vae_baseline.keras'):
                os.remove('models/vae_baseline.keras')
            #Basic baseline VAE: Fewer filters, simpler architecture
            encoder_baseline = create_baseline_encoder("baseline_vae")
            decoder_baseline = create_decoder(num_filters=32)
            vae_baseline = VAE(encoder_baseline, decoder_baseline)
            vae_baseline.compile(optimizer=keras.optimizers.Adam(learning_rate = 0.0001))

            #Train and save baseline VAE
            print("Training Baseline VAE")
            vae_baseline.fit(digits, epochs=10, batch_size=128, verbose=2)
            vae_baseline.save('models/vae_baseline.keras')

        if replace_big_model:
            if os.path.exists('models/vae_big.keras'):
                os.remove('models/vae_big.keras')
            #Teacher VAE: More filters, more complex architecture
            encoder_big = create_teacher_encoder()
            decoder_big = create_decoder(num_filters=32)
            vae_big =VAE(encoder_big, decoder_big)
            vae_big.compile(optimizer=keras.optimizers.Adam(learning_rate = 0.0001))

            print("Training Big VAE")
            vae_big.fit(digits, epochs=20, batch_size=128, verbose=2)

            vae_big.save('models/vae_big.keras')

        if replace_tc_model:
            if os.path.exists('models/tcvae_teacher.keras'):
                os.remove('models/tcvae_teacher.keras')
            #Teacher VAE: More filters, more complex architecture
            encoder_tc = create_teacher_encoder()
            decoder_tc = create_decoder(num_filters=32)
            vae_tc = TC_VAE(encoder_tc, decoder_tc, beta=1.0, beta_tc=0.0)
            vae_tc.compile(optimizer=keras.optimizers.Adam(learning_rate = 0.0001))

            #Add BetaScheduler
            beta_tc_var = tf.Variable(0.0, trainable=False, dtype=tf.float32)
            callbacks = [BetaScheduler(beta_tc_var, max_beta=1.0, schedule_epochs=10)]

            #Train
            print("Training Teacher VAE")
            vae_tc.fit(digits, epochs=20, batch_size=128, callbacks=callbacks, verbose=2)
            vae_tc.save('models/tcvae_teacher.keras')


        if replace_student_model:
            if os.path.exists('models/vae_student.keras'):
                os.remove('models/vae_student.keras')
            #Student VAE: Same size as baseline VAE, but trained using distillation
            teacher_vae = keras.models.load_model('models/vae_big.keras', custom_objects={'VAE': VAE})
            
            encoder_student = create_baseline_encoder("student_vae")
            decoder_student = create_decoder(num_filters=32)
            vae_student = DistillationVAE(encoder=encoder_student, decoder=decoder_student, teacher_model=teacher_vae, alpha=0.5)
            vae_student.compile(optimizer=keras.optimizers.Adam(learning_rate = 0.0001))
            print("Training Student VAE")
            vae_student.fit(digits, epochs=10, batch_size=128, verbose=2)

            vae_student.save('models/vae_student.keras')

        if replace_student_tc_model:
            if os.path.exists('models/vae_student_tc.keras'):
                os.remove('models/vae_student_tc.keras')
            #Student VAE: Same size as baseline VAE, but trained using distillation
            teacher_vae = keras.models.load_model('models/tcvae_teacher.keras', custom_objects={'TC_VAE': TC_VAE})
            
            encoder_student_tc = create_baseline_encoder("student_tc_vae")
            decoder_student_tc = create_decoder(num_filters=32)
            vae_student_tc = DistillationVAE(encoder=encoder_student_tc, decoder=decoder_student_tc, teacher_model=teacher_vae, alpha=0.5)
            vae_student_tc.compile(optimizer=keras.optimizers.Adam(learning_rate = 0.0001))
            print("Training Student TC VAE")
            vae_student_tc.fit(digits, epochs=10, batch_size=128, verbose=2)

            vae_student_tc.save('models/vae_student_tc.keras')
    finally:
        # Restore original stdout
        sys.stdout = sys.__stdout__
