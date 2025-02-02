import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
from sklearn.model_selection import train_test_split

prompts_jailbreaking = np.load("./activations/prompts_jailbreaking_after_intervention.npy", allow_pickle=True)
no_prompts_jailbreaking = np.load("./activations/no_prompts_jailbreaking_after_intervention.npy", allow_pickle=True)

data_jailbreak = prompts_jailbreaking.reshape(len(prompts_jailbreaking), 28, 3072)  
data_nonjailbreak = no_prompts_jailbreaking.reshape(len(no_prompts_jailbreaking), 28, 3072)


train_jailbreak, test_jailbreak = train_test_split(data_jailbreak, test_size=0.2, random_state=42)
train_nonjailbreak, test_nonjailbreak = train_test_split(data_nonjailbreak, test_size=0.2, random_state=42)

# Combine test datasets for t-SNE visualization
combined_test_data = np.concatenate([test_jailbreak, test_nonjailbreak], axis=0)

# Directory to save images
output_dir = Path("tsne_test_layer_images_intervention")
output_dir.mkdir(parents=True, exist_ok=True)

# Loop through each layer to generate t-SNE visualizations
for layer_idx in range(28):
    # Extract data for this layer (Shape: [n_samples, features])
    layer_data_jailbreak = test_jailbreak[:, layer_idx, :]  # Shape (samples, 3072)
    layer_data_nonjailbreak = test_nonjailbreak[:, layer_idx, :]  # Shape (samples, 3072)
    
    # Combine both datasets for the t-SNE
    combined_layer_data = np.concatenate([layer_data_jailbreak, layer_data_nonjailbreak], axis=0)
    
    # Apply t-SNE
    n_components = 2  # For 2D visualization
    tsne = TSNE(n_components=n_components, random_state=42)
    reduced_layer_data = tsne.fit_transform(combined_layer_data.reshape(-1, 3072))  # Shape: (test_samples, n_components)

    # Split the reduced data back into jailbreak and non-jailbreak
    reduced_layer_data_jailbreak = reduced_layer_data[:len(test_jailbreak), :]
    reduced_layer_data_nonjailbreak = reduced_layer_data[len(test_jailbreak):, :]

    # Create and save t-SNE plot for this layer
    plt.figure(figsize=(6, 4))
    plt.scatter(reduced_layer_data_jailbreak[:, 0], reduced_layer_data_jailbreak[:, 1], alpha=0.7, label='Jailbreak (Test)', color='cyan')
    plt.scatter(reduced_layer_data_nonjailbreak[:, 0], reduced_layer_data_nonjailbreak[:, 1], alpha=0.7, label='Non-Jailbreak (Test)', color='orange')

    plt.title(f"t-SNE Representation - Layer {layer_idx+1} (Test Data)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(loc="upper right")
    plt.grid(True)

    # Save the figure for this layer
    save_path = output_dir / f"tsne_test_layer_{layer_idx+1}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()  # Close the figure to free memory

    print(f"Saved t-SNE plot for Layer {layer_idx+1} at {save_path}")