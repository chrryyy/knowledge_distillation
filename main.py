import sys

from evaluation import (
    load_models,
    load_data,
    extract_latent_vectors,
    visualize_latent_space,
    calculate_and_plot_mig,
    reconstruction_accuracy,
)

#File paths
baseline_model_path = 'models/vae_baseline.keras'
student_model_path = 'models/vae_student.keras'
student_tc_model_path = 'models/vae_student_tc.keras'
big_model_path = 'models/vae_big.keras'
tc_model_path = 'models/tcvae_teacher.keras'
metrics_path = "dataset/t10k-morpho.csv"

#Load models and data
vae_baseline, vae_student, vae_student_tc, vae_big, vae_tc = load_models(baseline_model_path, student_model_path, student_tc_model_path, big_model_path, tc_model_path)
digits = load_data()

#Factors of variation
factors_cols = ["length", "thickness", "slant", "width", "height"]

# Specify the file to write output to
output_file = "output_logs/main.txt"

with open(output_file, "w") as f:
    sys.stdout = f
    try:
        #Calculate and visualize MIG
        print("baseline VAE MIG Calculation:")
        calculate_and_plot_mig(vae_baseline, "baseline_vae", metrics_path, digits, factors_cols)

        print("Student VAE MIG Calculation:")
        calculate_and_plot_mig(vae_student, "student_vae", metrics_path, digits, factors_cols)

        print("Student TC VAE MIG Calculation:")
        calculate_and_plot_mig(vae_student_tc, "student_tc_vae", metrics_path, digits, factors_cols)

        print("Big VAE MIG Calculation:")
        calculate_and_plot_mig(vae_big, "big_vae", metrics_path, digits, factors_cols)

        print("TC VAE MIG Calculation:")
        calculate_and_plot_mig(vae_tc, "tc_vae", metrics_path, digits, factors_cols)


        # Calculate and print reconstruction accuracies
        print("Reconstruction Accuracy:")

        baseline_accuracy = reconstruction_accuracy(vae_baseline, digits)
        print(f"Baseline VAE Reconstruction Accuracy: {baseline_accuracy}")

        student_accuracy = reconstruction_accuracy(vae_student, digits)
        print(f"Student VAE Reconstruction Accuracy: {student_accuracy}")

        student_tc_accuracy = reconstruction_accuracy(vae_student_tc, digits)
        print(f"Student TC VAE Reconstruction Accuracy: {student_tc_accuracy}")

        big_accuracy = reconstruction_accuracy(vae_big, digits)
        print(f"Big VAE Reconstruction Accuracy: {big_accuracy}")

        tc_accuracy = reconstruction_accuracy(vae_tc, digits)
        print(f"TC VAE Reconstruction Accuracy: {tc_accuracy}")
    finally:
        # Restore original stdout
        sys.stdout = sys.__stdout__