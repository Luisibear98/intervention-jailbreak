import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

# Load datasets
prompts_jailbreaking = np.load("./activations/prompts_jailbreaking_after_intervention.npy", allow_pickle=True)
no_prompts_jailbreaking = np.load("./activations/no_prompts_jailbreaking_after_intervention.npy", allow_pickle=True)

print(f"Jailbreaking samples: {len(prompts_jailbreaking)}")
print(f"Non-jailbreaking samples: {len(no_prompts_jailbreaking)}")

# Reshape to (samples, layers, features)
prompts_jailbreaking = np.array(prompts_jailbreaking).reshape(len(prompts_jailbreaking), 28, 3072)
no_prompts_jailbreaking = np.array(no_prompts_jailbreaking).reshape(len(no_prompts_jailbreaking), 28, 3072)

n_components = 2  # Number of PCA components

# Directory to save images
output_dir = Path("pca_layer_images_intervention")
output_dir.mkdir(parents=True, exist_ok=True)

# Loop through each layer
for layer_idx in range(28):
    # Extract layer-wise data
    layer_data_jailbreak = prompts_jailbreaking[:, layer_idx, :]  # (samples, features)
    layer_data_non_jailbreak = no_prompts_jailbreaking[:, layer_idx, :]  # (samples, features)

    # Apply PCA on combined data to have the same transformation space
    combined_data = np.vstack((layer_data_jailbreak, layer_data_non_jailbreak))  # (samples, features)
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(combined_data)  # (samples, 2)

    # Split back into jailbreak and non-jailbreak
    reduced_jailbreak = reduced_data[: len(layer_data_jailbreak)]
    reduced_non_jailbreak = reduced_data[len(layer_data_jailbreak) :]

    # Create and save PCA plot
    plt.figure(figsize=(6, 4))
    plt.scatter(reduced_jailbreak[:, 0], reduced_jailbreak[:, 1], alpha=0.7, label="Jailbreaking", color="red")
    plt.scatter(reduced_non_jailbreak[:, 0], reduced_non_jailbreak[:, 1], alpha=0.7, label="Non-Jailbreaking", color="blue")
    
    plt.title(f"PCA Representation - Layer {layer_idx+1}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)

    # Save the figure
    save_path = output_dir / f"pca_layer_intervention_{layer_idx+1}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()  # Free memory

    print(f"Saved PCA plot for Layer {layer_idx+1} at {save_path}")