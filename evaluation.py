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

def interpolate_and_plot_latent_space(models, model_names, img1, img2, num_steps=10, save_path="figures/interpolations.png"):
    #Linear interpolation between two images in latent space
    plt.figure(figsize=(15, len(models) * 2))

    for row_idx, (model, model_name) in enumerate(zip(models, model_names)):
        #Encode images into latent space
        z_mean1, _, _ = model.encoder.predict(img1)
        z_mean2, _, _ = model.encoder.predict(img2)

        #Linearly interpolate
        interpolated_images = []
        for alpha in np.linspace(0, 1, num_steps):
            z_interp = (1 - alpha) * z_mean1 + alpha * z_mean2
            reconstructed = model.decoder.predict(z_interp)
            interpolated_images.append(reconstructed[0, :, :, 0])

        #Add original images to the first and last step
        interpolated_images.insert(0, img1[0, :, :, 0])
        interpolated_images.append(img2[0, :, :, 0])

        #Plot all images
        for col_idx, img in enumerate(interpolated_images):
            plt.subplot(len(models), num_steps + 2, row_idx * (num_steps + 2) + col_idx + 1)
            plt.imshow(img, cmap="gray")
            plt.axis("off")
            if row_idx == 0:
                if col_idx == 0:
                    plt.title("Original 1")
                elif col_idx == len(interpolated_images) - 1:
                    plt.title("Original 2")
                else:
                    plt.title(f"Step {col_idx}")
        plt.suptitle(f"Latent Space Interpolation ({model_name})", y=1.02, fontsize=16)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_tangent_vectors_latent_axes(jacobian, image_shape, model_name, title_suffix=""):
    #Helper function for visualizing latent space tangent vectors
    num_inputs = jacobian.shape[1]
    ncols = 16
    nrows = int(np.ceil(num_inputs / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = axes.flatten()

    for i in range(num_inputs):
        tangent_vector = jacobian[:, i]
        image = tangent_vector.reshape(image_shape)
        #Normalize
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        axes[i].imshow(image, cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"Axis {i+1}", fontsize=6)

    #Hide unused subplots
    for i in range(num_inputs, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"{model_name} Latent Axes Tangent Vectors {title_suffix}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"figures/latent_tangent_vectors_{model_name}.png")

def plot_tangent_vectors_principal_components(jacobian, image_shape, model_name, num_components=128, title_suffix=""):
    #Helper function for visualizing principal component tangent vectors
    U, S, V = np.linalg.svd(jacobian, full_matrices=False)

    components_to_plot = min(num_components, V.shape[1])
    ncols = 16
    nrows = int(np.ceil(components_to_plot / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = axes.flatten()

    for i in range(components_to_plot):
        principal_tangent = jacobian @ V[i]
        image = principal_tangent.reshape(image_shape)
        #Normalize
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        axes[i].imshow(image, cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"PC {i+1}", fontsize=6)

    #Hide unused subplots
    for i in range(components_to_plot, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"{model_name} Principal Components Tangent Vectors {title_suffix}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"figures/latent_tangent_vectors_{model_name}.png")


def compute_and_plot_tangent_vectors(models, model_names, test_sample):
    #Compute and visualize tangent vectors in latent space
    image_shape = test_sample.shape[1:]
    for model, model_name in zip(models, model_names):
        #Latent vector
        z_mean, _, _ = model.encoder.predict(test_sample)

         #Compute Jacobian
        jacobian = np.zeros((np.prod(image_shape), z_mean.shape[1]))

        for i in range(z_mean.shape[1]):
            z_input = np.zeros_like(z_mean)
            z_input[0, i] = 1.0
            decoded_output = model.decoder.predict(z_input)
            jacobian[:, i] = decoded_output.flatten()

        #Plot tangent vectors
        plot_tangent_vectors_latent_axes(jacobian, image_shape, model_name)
        plot_tangent_vectors_principal_components(jacobian, image_shape, model_name)
