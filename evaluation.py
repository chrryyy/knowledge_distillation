from keras.models import load_model
from model_architecture import Sampling, BetaVAE, DistillationVAE
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
import seaborn as sns


def load_models(small_model_path, student_model_path, student_tc_model_path, big_model_path, tc_model_path):
    #Load pre-trained VAE models.
    vae_small = load_model(small_model_path, custom_objects={"Sampling": Sampling})
    vae_big = load_model(big_model_path, custom_objects={"Sampling": Sampling})
    vae_tc = load_model(tc_model_path, custom_objects={"Sampling": Sampling})
    vae_student = keras.models.load_model(student_model_path, custom_objects={"DistillationVAE": DistillationVAE,"teacher_model": vae_big, "Sampling": Sampling})
    vae_student.teacher_model = vae_big

    vae_student_tc = keras.models.load_model(student_tc_model_path, custom_objects={"DistillationVAE": DistillationVAE,"teacher_model": vae_tc, "Sampling": Sampling})
    vae_student_tc.teacher_model = vae_tc

    
    return vae_small, vae_student, vae_student_tc, vae_big, vae_tc


def load_data():
    #Load and normalize MNIST dataset.
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    digits = np.expand_dims(x_test, -1).astype("float32") / 255
    return digits


def extract_latent_vectors(vae_small, vae_big, data):
    #Extract latent vectors from the small and big VAEs.
    z_mean_small, _, _ = vae_small.encoder.predict(data)
    z_mean_big, _, _ = vae_big.encoder.predict(data)
    return z_mean_small, z_mean_big


def visualize_latent_space(z_mean_small, z_mean_big):
    #Apply PCA and visualize latent spaces.
    pca_small = PCA(n_components=2).fit_transform(z_mean_small)
    pca_big = PCA(n_components=2).fit_transform(z_mean_big)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(pca_small[:, 0], pca_small[:, 1], alpha=0.5)
    plt.title("Small VAE Latent Space")
    plt.subplot(1, 2, 2)
    plt.scatter(pca_big[:, 0], pca_big[:, 1], alpha=0.5)
    plt.title("Big VAE Latent Space")
    plt.show()


def discretize_data(data, n_bins=10):
    #Discretize continuous data into bins for MI calculation.
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    return discretizer.fit_transform(data)


def calculate_mi(latent_codes, factors):
    #Calculate mutual information (MI) between latent codes and factors.
    mi_matrix = np.zeros((latent_codes.shape[1], factors.shape[1]))
    for i in range(latent_codes.shape[1]):
        for j in range(factors.shape[1]):
            mi_matrix[i, j] = mutual_info_score(latent_codes[:, i], factors[:, j])
    return mi_matrix


def compute_mig(mi_matrix, factors):
   #Compute Mutual Information Gap (MIG) using MI matrix.
    entropy = np.array([mutual_info_score(f, f) for f in factors.T])
    mig_scores = []
    for j in range(mi_matrix.shape[1]):
        #Sort MI values for the factor
        sorted_mi = np.sort(mi_matrix[:, j])
        if len(sorted_mi) > 1:
            #Compute MIG as MI1 - MI2 / H(f)
            mig = (sorted_mi[-1] - sorted_mi[-2]) / entropy[j]
            mig_scores.append(mig)
    return np.mean(mig_scores), mig_scores


def calculate_and_plot_mig(models, model_names, metrics_path, digits, factors_cols):
    #Load metrics, calculate latent codes, and compute MIG

    #Load ground truth factors and latent codes
    metrics = pd.read_csv(metrics_path)
    factors = metrics[factors_cols].to_numpy()
    digits_subset = digits[-len(factors):]

    for model, model_name in zip(models, model_names):
        print(f"{model_name} MIG Calculation:")
        z_mean, _, _ = model.encoder.predict(digits_subset, verbose=2)

        #Discretize latent codes and factors for MI calculation
        discretized_latent_codes = discretize_data(z_mean, n_bins=10)
        discretized_factors = discretize_data(factors, n_bins=10)

        mi_matrix = calculate_mi(discretized_latent_codes, discretized_factors)
        mig_score, individual_mig_scores = compute_mig(mi_matrix, discretized_factors)

        print(f"MIG Score: {mig_score}")
        print(f"Individual MIG Scores: {individual_mig_scores}")

        #Heatmap of mutual information
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            mi_matrix,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            xticklabels=factors_cols,
            yticklabels=[f"z{i}" for i in range(z_mean.shape[1])],
        )
        plt.title(f"Mutual Information Between Latent Variables and Factors - {model_name}")
        plt.xlabel("Factors")
        plt.ylabel("Latent Variables")
        plt.savefig(f"figures/MI_heatmap_{model_name}.png")

#Reconstruction accuracy (General performance)
def reconstruction_accuracy(models, model_names, data):
    for model, model_name in zip(models, model_names):
        reconstructed = model.predict(data, verbose=2)
        mse = np.mean(np.square(data - reconstructed))
        accuracy = 1 - mse
        print(f"{model_name} Reconstruction Accuracy: {accuracy}")


def visualize_reconstruction(models, model_names, test_sample, save_path="figures/reconstructions.png"):
    reconstructed_images = [model.predict(test_sample, verbose=0) for model in models]

    plt.figure(figsize=(15, 4))

    #Plot original input
    plt.subplot(1, len(models) + 1, 1)
    plt.imshow(test_sample[0, :, :, 0], cmap="gray")
    plt.title("Original")
    plt.axis("off")

    #Plot reconstructions
    for i, (reconstruction, model_name) in enumerate(zip(reconstructed_images, model_names)):
        plt.subplot(1, len(models) + 1, i + 2)
        plt.imshow(reconstruction[0, :, :, 0], cmap="gray")
        plt.title(model_name)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()