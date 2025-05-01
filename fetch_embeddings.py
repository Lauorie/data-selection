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

# texts = ["第一句话。", "这是第二句，它更长一些。"]
def get_embeddings(texts, model, tokenizer, device, batch_size=32, max_length=512):
    """
    获取文本列表的 embeddings。
    ... (函数文档字符串省略) ...
    """
    # 1. 初始化一个列表，用来收集所有批次处理后的 embedding 结果
    all_embeddings = []
    # 确保模型在正确的设备上 (如果用了 device_map='auto'，这步可能不是必须的，但显式指定无害)
    # model.to(device) # 如果 device_map='auto'，模型可能已分布在多设备

    # 2. 批处理循环：为了高效处理大量数据并控制内存使用，我们分批处理文本
    # tqdm(...) 提供一个进度条
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
        # 获取当前批次的文本
        batch_texts = texts[i:i+batch_size]

        # 3. 分词 (Tokenization): 将文本字符串转换为模型能理解的数字 ID
        # 这是非常关键的一步
        inputs = tokenizer(
            batch_texts,          # 输入当前批次的文本列表
            padding=True,         # <--- 重要：填充 (Padding)
                                  #      使得这个批次内的所有序列都具有相同的长度
                                  #      (等于批次中最长序列的长度)。
                                  #      用特殊的 padding token ID 来填充。
            truncation=True,      # <--- 重要：截断 (Truncation)
                                  #      如果序列超过 max_length，就把它截断。
                                  #      防止序列过长导致内存溢出或超出模型处理能力。
            max_length=max_length,  # 指定最大序列长度
            return_tensors="pt"   # 返回 PyTorch 张量 (Tensor)
        ).to(device) # 将生成的张量移动到指定设备 (CPU 或 GPU)

        # `inputs` 现在是一个字典，通常包含：
        # - 'input_ids': 形状是 (batch_size, sequence_length)，每个数字是 token 的 ID。
        # - 'attention_mask': 形状是 (batch_size, sequence_length)，
        #                    值为 1 的位置是真实的 token，值为 0 的位置是 padding token。
        #                    这个 mask 非常重要，后续会用它来区分真实内容和填充物。
        # (可能还有 'token_type_ids' 等，取决于模型)

        # 4. 模型推理 (Inference): 将 token 输入模型，获取输出
        # `torch.no_grad()` 表示我们不需要计算梯度 (因为不是在训练)，可以节省计算和内存
        with torch.no_grad():
            # 将分词后的输入 (`inputs`) 传入模型
            # `**inputs` 是 Python 的解包语法，相当于 model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], ...)
            outputs = model(**inputs, output_hidden_states=False)

            # `outputs` 是一个包含模型输出的对象。我们最关心的是 `last_hidden_state`。
            # `last_hidden_state` 的形状是: (batch_size, sequence_length, hidden_size)
            # - batch_size: 当前批次中的句子数量。
            # - sequence_length: 经过 padding/truncation 后，这个批次中所有序列的统一长度。
            # - hidden_size: 模型隐藏层的大小 (例如 768, 1024, 4096 等)。这是每个 token 的向量维度。
            #
            # !!! 理解关键点 !!!
            # 到这里，模型为 *每个句子* 中的 *每个 token* (包括 padding token) 都生成了一个 `hidden_size` 维度的向量。
            # 例如，如果 batch_size=2, sequence_length=10, hidden_size=768，
            # 那么 `last_hidden_state` 的形状就是 (2, 10, 768)。
            # 我们现在有 2 个句子的 token 向量表示，但每个句子是由 10 个 768 维的向量组成的。
            # 我们的目标是：为每个句子生成 *一个* 768 维的向量。

        # 5. 池化操作 (Pooling): 从每个序列的多个 token 向量中计算出一个单一的序列向量
        #    这里我们使用 Mean Pooling (平均池化) 策略。
        #    目标：计算每个句子所有 *非填充* token 的 hidden state 的平均值。

        # 5a. 获取最后一层的隐藏状态
        last_hidden_states = outputs.last_hidden_state # Shape: (batch_size, sequence_length, hidden_size)

        # 5b. 获取 Attention Mask，它告诉我们哪些是真实 token (1)，哪些是 padding (0)
        attention_mask = inputs['attention_mask']      # Shape: (batch_size, sequence_length)

        # 5c. 为了利用 attention_mask 来屏蔽 padding token 的 hidden state，我们需要扩展 mask 的维度
        #     unsqueeze(-1) 在最后增加一个维度: (batch_size, sequence_length) -> (batch_size, sequence_length, 1)
        #     expand(...) 将最后一个维度扩展到 hidden_size: (batch_size, sequence_length, 1) -> (batch_size, sequence_length, hidden_size)
        #     这样 mask 就和 last_hidden_states 的形状一样了。
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()

        # 5d. 将 padding token 的 hidden state 置零
        #     通过逐元素相乘，真实 token 的 hidden state 乘以 1 (不变)，padding token 的 hidden state 乘以 0 (变为零)。
        masked_hidden_states = last_hidden_states * input_mask_expanded # Shape: (batch_size, sequence_length, hidden_size)

        # 5e. 计算每个序列的 *有效* hidden state 的总和
        #     torch.sum(..., 1) 在 sequence_length 维度上求和。
        #     结果 sum_embeddings 的形状是 (batch_size, hidden_size)。
        #     现在，每个句子的向量是其所有非填充 token 向量的和。
        sum_embeddings = torch.sum(masked_hidden_states, 1)

        # 5f. 计算每个序列中 *有效* token 的数量
        #     我们直接对 expanded mask 在 sequence_length 维度上求和。
        #     注意：这里求和的是 mask (0 或 1)，所以结果是每个序列中 1 的数量，即有效 token 的数量。
        #     clamp(min=1e-9) 是为了防止除以零（如果一个序列全是 padding，虽然不太可能）。
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9) # Shape: (batch_size, hidden_size)

        # 5g. 计算平均值：用总和除以数量
        #     这得到了每个句子的 Mean Pooling Embedding。
        mean_pooled_embeddings = sum_embeddings / sum_mask # Shape: (batch_size, hidden_size)
        # !!! 理解关键点 !!!
        # 经过这步 Mean Pooling，我们成功地将每个句子的表示从 (sequence_length, hidden_size) 压缩成了 (hidden_size)。
        # `mean_pooled_embeddings` 现在包含了当前批次中每个句子的单一向量表示。

        # 6. 收集结果
        #    将计算得到的当前批次的 embeddings (PyTorch Tensor) 转移到 CPU，
        #    并转换为 NumPy 数组 (更通用的格式，方便后续处理如 scikit-learn 聚类)，
        #    然后添加到 `all_embeddings` 列表中。
        all_embeddings.append(mean_pooled_embeddings.cpu().numpy())

    # 7. 合并所有批次的结果
    #    当循环结束后，`all_embeddings` 是一个包含多个 NumPy 数组 (每个代表一个批次) 的列表。
    #    `np.concatenate(..., axis=0)` 将这些数组沿着第一个维度 (批次维度) 拼接起来，
    #    形成一个大的 NumPy 数组，形状为 (总文本数量, hidden_size)。
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
