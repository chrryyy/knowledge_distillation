from evaluation import (
    load_models,
    load_data,
    extract_latent_vectors,
    visualize_latent_space,
    calculate_and_plot_mig,
)

#File paths
baseline_model_path = 'models/vae_baseline.keras'
big_model_path = 'models/vae_teacher.keras'
teacher_model_path = 'models/bvae_teacher.keras'
tc_model_path = 'models/tcvae_teacher.keras'
metrics_path = "dataset/t10k-morpho.csv"

#Load models and data
vae_baseline, vae_big, vae_teacher, vae_tc = load_models(baseline_model_path, big_model_path, teacher_model_path, tc_model_path)
digits = load_data()

#Factors of variation
factors_cols = ["length", "thickness", "slant", "width", "height"]

#Calculate and visualize MIG
print("baseline VAE MIG Calculation:")
calculate_and_plot_mig(vae_baseline, metrics_path, digits, factors_cols)

print("Big VAE MIG Calculation:")
calculate_and_plot_mig(vae_big, metrics_path, digits, factors_cols)

print("Teacher VAE MIG Calculation:")
calculate_and_plot_mig(vae_teacher, metrics_path, digits, factors_cols)

print("TC VAE MIG Calculation:")
calculate_and_plot_mig(vae_tc, metrics_path, digits, factors_cols)
