# SFT 数据 Embedding、聚类与可视化

该项目包含两个 Python 脚本，用于处理和分析 SFT (Supervised Fine-Tuning) 数据集中的用户查询：

1.  **`fetch_embeddings.py`**: 使用预训练的 Transformer 模型为 SFT 数据中的用户查询生成文本 embedding，然后对这些 embedding 进行 K-Means 聚类，并将结果（embedding、聚类标签、聚类中心、带标签的原始数据）保存到指定目录。
2.  **`visualize_embeddings.py`**: 加载由 `fetch_embeddings.py` 生成的 embedding 和聚类标签，使用 t-SNE 或 UMAP 进行降维，并将降维后的 embedding 绘制成散点图进行可视化，图中点的颜色代表其所属的聚类。

## 功能

*   从 JSON 文件加载 SFT 数据。
*   使用 Hugging Face `transformers` 库加载指定的预训练模型和分词器。
*   高效地批量生成文本 embedding。
*   支持加载预先计算好的 embedding 文件。
*   使用 `scikit-learn` 对 embedding 进行 K-Means 聚类。
*   将 embedding、聚类标签、聚类中心以及带有聚类标签的原始数据保存为文件。
*   使用 t-SNE 或 UMAP 对高维 embedding 进行降维。
*   使用 `matplotlib` 和 `seaborn` 生成美观的 2D 散点图可视化聚类结果。
*   灵活的命令行参数配置。

## 依赖库

运行这些脚本需要安装以下 Python 库：

*   `torch`
*   `transformers`
*   `numpy`
*   `tqdm`
*   `scikit-learn`
*   `matplotlib`
*   `seaborn`
*   `umap-learn` (如果需要使用 UMAP 可视化)

建议使用 `pip` 创建虚拟环境并安装依赖：

```bash
pip install torch transformers numpy tqdm scikit-learn matplotlib seaborn umap-learn
```

## 使用方法

### 1. 生成 Embedding 和聚类

运行 `fetch_embeddings.py` 脚本。你需要提供模型名称/路径、SFT 数据 JSON 文件路径以及输出目录。

**输入数据格式:**

脚本期望输入的 JSON 文件包含一个列表，列表中的每个元素是一个字典，代表一个对话样本。脚本会提取每个样本中 `"conversations"` 列表里第一个元素的 `"value"` 字段作为用户查询文本。例如：

```json
[
  {
    "id": "identity_0",
    "conversations": [
      {
        "from": "user",
        "value": "这是第一个用户查询。"
      },
      {
        "from": "assistant",
        "value": "这是助手的回答。"
      }
    ]
  },
  {
    "id": "identity_1",
    "conversations": [
      {
        "from": "user",
        "value": "这是第二个用户查询。"
      },
      {
        "from": "assistant",
        "value": "这是另一个回答。"
      }
    ]
  }
]
```

**示例命令:**

```bash
python data-clean/src/fetch_embeddings.py \
    --model_name_or_path <your_model_name_or_path> \
    --json_path <path_to_your_sft_data.json> \
    --output_dir cluster_results \
    --n_clusters 10 \
    --batch_size 32 \
    --max_length 512
```

*   将 `<your_model_name_or_path>` 替换为你选择的 Hugging Face 模型名称（例如 `bert-base-uncased`）或本地模型路径。
*   将 `<path_to_your_sft_data.json>` 替换为你的 SFT 数据文件路径。
*   `--output_dir` 指定保存结果的目录（例如 `cluster_results`）。
*   可以通过 `--n_clusters`, `--batch_size`, `--max_length` 等参数调整聚类数量、批处理大小和最大序列长度。
*   如果已有计算好的 embedding 文件 (`.npy` 格式)，可以使用 `--embeddings_path` 参数指定路径以跳过 embedding 生成步骤。

**输出文件 (在 `output_dir` 中):**

*   `embeddings.npy`: 计算出的文本 embedding (NumPy 数组)。
*   `cluster_labels.npy`: 每个 embedding 对应的聚类标签 (NumPy 数组)。
*   `cluster_centers.npy`: K-Means 聚类中心 (NumPy 数组)。
*   `labeled_data.json`: 原始 SFT 数据，并为每个样本添加了 `cluster_label` 字段。

### 2. 可视化 Embedding

运行 `visualize_embeddings.py` 脚本。你需要提供上一步生成的 embedding 和聚类标签文件的路径。

**示例命令:**

*   **使用 t-SNE (默认):**
    ```bash
    python data-clean/src/visualize_embeddings.py \
        --embeddings_path cluster_results/embeddings.npy \
        --labels_path cluster_results/cluster_labels.npy \
        --output_image cluster_results/embeddings_tsne.png \
        --method tsne
    ```

*   **使用 UMAP:**
    ```bash
    python data-clean/src/visualize_embeddings.py \
        --embeddings_path cluster_results/embeddings.npy \
        --labels_path cluster_results/cluster_labels.npy \
        --output_image cluster_results/embeddings_umap.png \
        --method umap
    ```

*   `--embeddings_path` 和 `--labels_path` 应指向 `fetch_embeddings.py` 生成的对应文件。
*   `--output_image` 指定保存可视化图像的文件名。
*   使用 `--method` 选择降维方法 (`tsne` 或 `umap`)。
*   可以调整 t-SNE 的 `--perplexity` 或 UMAP 的 `--n_neighbors` 和 `--min_dist` 参数。

**输出文件:**

*   一个图像文件 (例如 `embeddings_tsne.png` 或 `embeddings_umap.png`)，展示了降维后的 embedding 散点图，点按聚类标签着色。

## 配置参数

两个脚本都提供了多个命令行参数来自定义其行为。使用 `-h` 或 `--help` 查看详细信息：

```bash
python data-clean/src/fetch_embeddings.py --help
python data-clean/src/visualize_embeddings.py --help
```
```
