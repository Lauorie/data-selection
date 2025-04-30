import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
# import umap # Or use UMAP
import os
import argparse

def visualize_embeddings(embeddings_path, labels_path, output_image_path, method='tsne', perplexity=30, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Loads embeddings and cluster labels, performs dimensionality reduction (t-SNE or UMAP),
    and saves a visualization plot.

    Args:
        embeddings_path (str): Path to the embeddings .npy file.
        labels_path (str): Path to the cluster labels .npy file.
        output_image_path (str): Path to save the output plot image (e.g., 'embeddings_tsne.png').
        method (str): Dimensionality reduction method ('tsne' or 'umap').
        perplexity (int): Perplexity parameter for t-SNE.
        n_neighbors (int): Number of neighbors parameter for UMAP.
        min_dist (float): Minimum distance parameter for UMAP.
        random_state (int): Random state for reproducibility.
    """
    print(f"Loading embeddings from: {embeddings_path}")
    try:
        embeddings = np.load(embeddings_path)
    except FileNotFoundError:
        print(f"Error: Embeddings file not found at {embeddings_path}")
        return
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return

    print(f"Loading cluster labels from: {labels_path}")
    try:
        cluster_labels = np.load(labels_path)
    except FileNotFoundError:
        print(f"Error: Cluster labels file not found at {labels_path}")
        # Allow visualization without labels if file is missing
        cluster_labels = None
        print("Proceeding without cluster labels.")
    except Exception as e:
        print(f"Error loading cluster labels: {e}")
        return

    if cluster_labels is not None and embeddings.shape[0] != cluster_labels.shape[0]:
        print(f"Warning: Embeddings count ({embeddings.shape[0]}) doesn't match label count ({cluster_labels.shape[0]}). Cannot use labels for coloring.")
        cluster_labels = None

    n_samples = embeddings.shape[0]
    print(f"Found {n_samples} embeddings.")

    # --- Dimensionality Reduction ---
    print(f"Performing {method.upper()} dimensionality reduction...")
    if method == 'tsne':
        # Adjust perplexity if number of samples is small
        actual_perplexity = min(perplexity, n_samples - 1)
        if actual_perplexity != perplexity:
             print(f"Warning: Perplexity ({perplexity}) is too large for {n_samples} samples. Using perplexity={actual_perplexity}.")
        if actual_perplexity <= 0:
             print("Error: Not enough samples for t-SNE (perplexity must be > 0).")
             return

        tsne = TSNE(n_components=2, random_state=random_state, perplexity=actual_perplexity, n_iter=300, init='pca')
        embeddings_2d = tsne.fit_transform(embeddings)
        plot_title = f't-SNE Visualization of Embeddings (Perplexity={actual_perplexity})'
    elif method == 'umap':
        try:
            import umap # Import UMAP here
            reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=n_neighbors, min_dist=min_dist)
            embeddings_2d = reducer.fit_transform(embeddings)
            plot_title = f'UMAP Visualization of Embeddings (n_neighbors={n_neighbors}, min_dist={min_dist})'
        except ImportError:
            print("Error: umap-learn library not installed. Please install it: pip install umap-learn")
            print("Falling back to t-SNE.")
            # Fallback to t-SNE
            actual_perplexity = min(perplexity, n_samples - 1)
            if actual_perplexity != perplexity:
                 print(f"Warning: Perplexity ({perplexity}) is too large for {n_samples} samples. Using perplexity={actual_perplexity}.")
            if actual_perplexity <= 0:
                 print("Error: Not enough samples for t-SNE (perplexity must be > 0).")
                 return
            tsne = TSNE(n_components=2, random_state=random_state, perplexity=actual_perplexity, n_iter=300, init='pca')
            embeddings_2d = tsne.fit_transform(embeddings)
            plot_title = f't-SNE Visualization of Embeddings (Fallback, Perplexity={actual_perplexity})'
    else:
        print(f"Error: Unknown dimensionality reduction method '{method}'. Choose 'tsne' or 'umap'.")
        return

    print("Dimensionality reduction complete.")

    # --- Plotting ---
    print("Generating plot...")
    plt.figure(figsize=(12, 10))

    # Determine unique labels and generate a color palette if labels exist
    unique_labels = np.unique(cluster_labels) if cluster_labels is not None else [0]
    n_clusters = len(unique_labels)
    palette = sns.color_palette("hsv", n_clusters) # 'hsv', 'viridis', 'tab10' are good options

    # Create scatter plot
    scatter = sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=cluster_labels if cluster_labels is not None else None, # Color by cluster label
        palette=palette if cluster_labels is not None else None,
        legend='full' if cluster_labels is not None else False,
        alpha=0.7, # Adjust transparency
        s=50 # Adjust point size
    )

    plt.title(plot_title)
    plt.xlabel(f"{method.upper()} Dimension 1")
    plt.ylabel(f"{method.upper()} Dimension 2")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Improve legend if there are many clusters
    if cluster_labels is not None and n_clusters > 10:
        # Place legend outside the plot
        scatter.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    else:
        plt.tight_layout()


    # Save the plot
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight') # Save with high resolution
    print(f"Plot saved to {output_image_path}")
    # Optionally show the plot
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize text embeddings using t-SNE or UMAP.")
    parser.add_argument("--embeddings_path", type=str, default="cluster_results/embeddings.npy", help="Path to the embeddings .npy file.")
    parser.add_argument("--labels_path", type=str, default="cluster_results/cluster_labels.npy", help="Path to the cluster labels .npy file.")
    parser.add_argument("--output_image", type=str, default="cluster_results/embeddings_visualization.png", help="Path to save the output plot image.")
    parser.add_argument("--method", type=str, default="tsne", choices=['tsne', 'umap'], help="Dimensionality reduction method (tsne or umap).")
    # t-SNE specific args
    parser.add_argument("--perplexity", type=int, default=30, help="Perplexity for t-SNE.")
    # UMAP specific args
    parser.add_argument("--n_neighbors", type=int, default=15, help="Number of neighbors for UMAP.")
    parser.add_argument("--min_dist", type=float, default=0.1, help="Minimum distance for UMAP.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility.")

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_image)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    visualize_embeddings(
        args.embeddings_path,
        args.labels_path,
        args.output_image,
        method=args.method,
        perplexity=args.perplexity,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state
    )

# 使用 t-SNE (默认)
# python visualize_embeddings.py --embeddings_path cluster_results/embeddings.npy --labels_path cluster_results/cluster_labels.npy --output_image cluster_results/embeddings_tsne.png

# 使用 UMAP
# python visualize_embeddings.py --method umap --embeddings_path cluster_results/embeddings.npy --labels_path cluster_results/cluster_labels.npy --output_image cluster_results/embeddings_umap.png
