import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from tqdm.auto import tqdm
import json
import argparse
import os
from sklearn.cluster import KMeans

# --- Helper Functions ---

def load_sft_data(json_path):
    """Loads SFT data and extracts the first user message."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            sft_data = json.load(f)
        # Extract the first turn's 'value' which is typically the user query
        texts = [item["conversations"][0]["value"] for item in sft_data if item["conversations"]]
        print(f"Loaded {len(texts)} texts from {json_path}")
        return texts, sft_data # Return original data as well for saving later
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        exit(1)
    except Exception as e:
        print(f"Error loading or parsing JSON: {e}")
        exit(1)

def setup_model_and_tokenizer(model_name_or_path, device):
    """Loads the model and tokenizer."""
    print(f"Loading tokenizer for {model_name_or_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        exit(1)

    print(f"Loading model {model_name_or_path}...")
    try:
        # Use AutoModel to load the base model
        model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map='auto' # Automatically handle device placement
        )
        model.eval() # Set to evaluation mode
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    # Handle padding token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            # Check if model config needs update (might not be necessary with device_map='auto')
            # Only update if pad_token_id is not already set or different
            if model.config.pad_token_id is None or model.config.pad_token_id != tokenizer.eos_token_id:
                 try:
                     model.config.pad_token_id = tokenizer.eos_token_id
                     print("Set model.config.pad_token_id to eos_token_id.")
                 except AttributeError as e:
                     print(f"Warning: Could not set pad_token_id on model config: {e}")

            print("Set pad_token to eos_token.")
        else:
            print("Error: Model has neither pad_token nor eos_token. Cannot set padding.")
            # Decide how to handle this case, e.g., add a new pad token or exit
            # Adding a new token might require resizing model embeddings
            # For now, we'll exit.
            exit(1)


    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

def get_embeddings(texts, model, tokenizer, device, batch_size=32, max_length=512):
    """Generates embeddings for a list of texts using mean pooling."""
    all_embeddings = []
    model.eval() # Ensure model is in eval mode

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device) # Move inputs to the correct device

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=False)

        # Mean Pooling Calculation
        last_hidden_states = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled_embeddings = sum_embeddings / sum_mask

        all_embeddings.append(mean_pooled_embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)

def perform_clustering(embeddings, n_clusters, random_state=0):
    """Performs K-Means clustering."""
    print(f"Performing K-Means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings) # Use fit_predict
    print("Clustering complete.")
    return cluster_labels, kmeans.cluster_centers_

def save_results(output_dir, embeddings, cluster_labels, centers, original_data, texts_to_embed):
    """Saves embeddings, cluster labels, centers, and labeled data."""
    os.makedirs(output_dir, exist_ok=True)

    # Save embeddings
    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    np.save(embeddings_path, embeddings)
    print(f"Embeddings saved to {embeddings_path}")

    # Save cluster labels
    labels_path = os.path.join(output_dir, "cluster_labels.npy")
    np.save(labels_path, cluster_labels)
    print(f"Cluster labels saved to {labels_path}")

    # Save cluster centers
    centers_path = os.path.join(output_dir, "cluster_centers.npy")
    np.save(centers_path, centers)
    print(f"Cluster centers saved to {centers_path}")

    # Save original data with cluster labels
    labeled_data_path = os.path.join(output_dir, "labeled_data.json")
    labeled_data = []
    for i, item in enumerate(original_data):
         # Ensure we only label items that were actually embedded
        if i < len(texts_to_embed):
             # Find the original item corresponding to the text embedded
             # This assumes the order is preserved and texts_to_embed corresponds 1:1 with the start of original_data
            original_item = original_data[i]
            # Add the cluster label to the original item's data
            # You might want to add it differently, e.g., inside a 'metadata' field
            labeled_item = original_item.copy() # Avoid modifying original dict in place
            labeled_item['cluster_label'] = int(cluster_labels[i]) # Convert numpy int to standard int for JSON
            labeled_data.append(labeled_item)
        else:
             print(f"Warning: Index {i} out of bounds for texts_to_embed (length {len(texts_to_embed)}). Skipping labeling for this item.")


    with open(labeled_data_path, "w", encoding="utf-8") as f:
        json.dump(labeled_data, f, ensure_ascii=False, indent=4)
    print(f"Original data with cluster labels saved to {labeled_data_path}")


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for SFT data and perform clustering.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path or ID of the Hugging Face model.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the SFT JSON data file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save embeddings and clustering results.")
    parser.add_argument("--embeddings_path", type=str, default=None, help="Optional path to load pre-computed embeddings (.npy).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for embedding generation.")
    parser.add_argument("--max_length", type=int, default=4096, help="Max sequence length for tokenizer.")
    parser.add_argument("--n_clusters", type=int, default=10, help="Number of clusters for K-Means.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for K-Means.")
    parser.add_argument("--no_cuda", action='store_true', help="Disable CUDA even if available.")

    args = parser.parse_args()

    # --- Setup Device ---
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # --- Load Data ---
    texts_to_embed, original_sft_data = load_sft_data(args.json_path)
    if not texts_to_embed:
        print("No texts found to embed. Exiting.")
        return

    embeddings = None
    # --- Load or Generate Embeddings ---
    if args.embeddings_path and os.path.exists(args.embeddings_path):
        print(f"Loading pre-computed embeddings from {args.embeddings_path}...")
        try:
            embeddings = np.load(args.embeddings_path)
            print(f"Embeddings loaded. Shape: {embeddings.shape}")
            # Basic validation
            if embeddings.shape[0] != len(texts_to_embed):
                print(f"Warning: Loaded embeddings count ({embeddings.shape[0]}) doesn't match text count ({len(texts_to_embed)}). Recalculating embeddings.")
                embeddings = None # Force recalculation
        except Exception as e:
            print(f"Error loading embeddings: {e}. Recalculating embeddings.")
            embeddings = None # Force recalculation

    if embeddings is None:
        # --- Load Model and Tokenizer ---
        model, tokenizer = setup_model_and_tokenizer(args.model_name_or_path, device)

        # --- Generate Embeddings ---
        print("Starting embedding generation...")
        embeddings = get_embeddings(
            texts_to_embed,
            model,
            tokenizer,
            device,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
        print(f"Embeddings generated. Shape: {embeddings.shape}")

        # --- Clean up GPU memory ---
        del model
        del tokenizer
        if use_cuda:
            torch.cuda.empty_cache()
            print("Cleaned up GPU memory.")

    # --- Perform Clustering ---
    if embeddings is not None and embeddings.shape[0] > 0:
        cluster_labels, cluster_centers = perform_clustering(
            embeddings,
            n_clusters=args.n_clusters,
            random_state=args.random_state
        )

        # --- Save Results ---
        save_results(args.output_dir, embeddings, cluster_labels, cluster_centers, original_sft_data, texts_to_embed)

        print("\n--- Next Steps ---")
        print(f"Results saved in: {args.output_dir}")
        print("Consider analyzing the `labeled_data.json` file to inspect the content of each cluster.")
        print(f"You can also try different values for `--n_clusters` or explore other clustering algorithms.")
        print("Visualizing the embeddings (e.g., using t-SNE or UMAP) might also provide insights.")

    else:
        print("No embeddings available to perform clustering.")


if __name__ == "__main__":
    main()
