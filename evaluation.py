from keras.models import load_model
from model_architecture import Sampling, BetaVAE
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
import seaborn as sns


def load_models(small_model_path, big_model_path, teacher_model_path, tc_model_path):
    #Load pre-trained VAE models.
    vae_small = load_model(small_model_path, custom_objects={"Sampling": Sampling})
    vae_big = load_model(big_model_path, custom_objects={"Sampling": Sampling})
    vae_teacher = load_model(teacher_model_path, custom_objects={"Sampling": Sampling})
    vae_tc = load_model(tc_model_path, custom_objects={"Sampling": Sampling})
    return vae_small, vae_big, vae_teacher, vae_tc


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


def calculate_and_plot_mig(vae, metrics_path, digits, factors_cols):
    #Load metrics, calculate latent codes, and compute MIG

    #Load ground truth factors and latent codes
    metrics = pd.read_csv(metrics_path)
    factors = metrics[factors_cols].to_numpy()
    digits_subset = digits[-len(factors):]
    z_mean, _, _ = vae.encoder.predict(digits_subset)

    #Discretize latent codes and factors
    discretized_latent_codes = discretize_data(z_mean, n_bins=10)
    discretized_factors = discretize_data(factors, n_bins=10)

    #Calculate MI and MIG
    mi_matrix = calculate_mi(discretized_latent_codes, discretized_factors)
    mig_score, individual_mig_scores = compute_mig(mi_matrix, discretized_factors)

    #Print and plot results
    print("MIG Score:", mig_score)
    print("Individual MIG Scores:", individual_mig_scores)

    #Heatmap of mutual information
    plt.figure(figsize=(10, 8))
    sns.heatmap(mi_matrix, annot=True, fmt=".2f", cmap="viridis", xticklabels=factors_cols, yticklabels=[f"z{i}" for i in range(z_mean.shape[1])])
    plt.title("Mutual Information Between Latent Variables and Factors")
    plt.xlabel("Factors")
    plt.ylabel("Latent Variables")
    #Save to figures folder
    plt.savefig("figures/mutual_information_heatmap.png")

    return mig_score, mi_matrix
