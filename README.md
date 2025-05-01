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

## 重点代码理解
```python
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from tqdm.auto import tqdm # 用于显示进度条

# (假设 model, tokenizer, device 已经像之前那样加载好了)
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
```

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
