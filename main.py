import sys

from evaluation import (
    load_models,
    load_data,
    extract_latent_vectors,
    visualize_latent_space,
    calculate_and_plot_mig,
    reconstruction_accuracy,
    visualize_reconstruction,
)

#File paths
baseline_model_path = 'models/vae_baseline.keras'
student_model_path = 'models/vae_student.keras'
student_tc_model_path = 'models/vae_student_tc.keras'
big_model_path = 'models/vae_big.keras'
tc_model_path = 'models/tcvae_teacher.keras'
metrics_path = "dataset/t10k-morpho.csv"
output_file = "output_logs/main.txt"

#Load models and data
vae_baseline, vae_student, vae_student_tc, vae_big, vae_tc = load_models(baseline_model_path, student_model_path, student_tc_model_path, big_model_path, tc_model_path)
digits = load_data()

#Factors of variation
factors_cols = ["length", "thickness", "slant", "width", "height"]

# Models and their names
models = [vae_baseline, vae_student, vae_student_tc, vae_big, vae_tc]
model_names = ["Baseline VAE", "Student VAE", "Student TC VAE", "Big VAE", "TC VAE"]



with open(output_file, "w") as f:
    sys.stdout = f
    try:
        #Calculate and visualize MIG for all models
        calculate_and_plot_mig(models, model_names, metrics_path, digits, factors_cols)

        #Calculate reconstruction accuracy for all models
        reconstruction_accuracy(models, model_names, digits)

        #Visualize reconstructions for sample 1
        test_index = 1
        test_sample = digits[test_index:test_index + 1]
        visualize_reconstruction(models, model_names, test_sample, save_path="figures/reconstructions.png")

    finally:
        # Restore original stdout
        sys.stdout = sys.__stdout__