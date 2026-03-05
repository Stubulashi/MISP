
# %%
! pip install transformer_lens

# %%
import os
# EN: Set the system environment variable to ensure UTF-8 encoding for input/output.
# CN: 设置系统环境变量，确保输入输出使用 UTF-8 编码。
os.environ["PYTHONIOENCODING"] = "utf-8"

from transformer_lens import HookedTransformer

# %%
# EN: Load the pre-trained GPT-2 small model using the TransformerLens library.
# CN: 使用 TransformerLens 库加载预训练的 GPT-2 small 模型。
model = HookedTransformer.from_pretrained("gpt2-small")

# EN: Define a list of test words to observe how the tokenizer splits them.
# CN: 定义一组测试单词，用于观察分词器如何对其进行拆分。
test_words = ["unhappy", "unclear", "understand", "unique", "uncle", "unit"]

# EN: Print the table header for the tokenization results.
# CN: 打印分词结果的表格表头。
print(f"{'Word':<15} | {'Tokens':<30} | {'Split Check'}")
print("-" * 60)

for word in test_words:
    # EN: Convert the word into a list of string-formatted tokens.
    # CN: 将单词转换为字符串格式的 token 列表。
    tokens = model.to_str_tokens(word)

    # EN: Check if the word is split into multiple tokens (more than 2 including the BOS token).
    # CN: 检查单词是否被拆分为多个 token（计入起始符后长度大于 2 则视为被拆分）。
    is_split = "Yes" if len(tokens) > 2 else "No (Single Token)"
    print(f"{word:<15} | {str(tokens):<30} | {is_split}")

# %% [markdown]
# Here we can see that in the tokenizer, “understand” is split apart, while the other three words are treated as a single unit.

# %%
def find_split_traps(vocab_limit=20000):
    """
    EN: Identify words starting with 'un' that are tokenized specifically as ['un', ...].
    CN: 寻找以 'un' 开头，且被分词器切分为 ['un', '...'] 的单词。
    """
    traps = []
    # EN: List of potential words that might be split in unexpected ways by the tokenizer.
    # CN: 可能被分词器以非预期方式拆分的候选单词列表。
    candidates = [
        "union", "university", "universe", "unison", "unique",
        "unify", "unicorn", "uniform", "universal", "under", "uncle"
    ]

    print(f"{'Word':<15} | {'Tokens':<30} | {'Is Good Trap?'}")
    print("-" * 60)

    for word in candidates:
        tokens = model.to_str_tokens(word)
        # EN: Check if the first actual token (index 1) of the word is exactly 'un'.
        # CN: 检查单词的第一个实际 token（索引为 1）是否正是 'un'。
        if tokens[1] == "un": # tokens[0] 通常是 <|endoftext|>
            traps.append(word)
            status = "Marked"
        else:
            status = "No"

        print(f"{word:<15} | {str(tokens):<30} | {status}")

    return traps

# EN: Execute the search function and store the identified split traps.
# CN: 运行搜索函数并存储找到的分词陷阱单词。
found_traps = find_split_traps()

# %% [markdown]
# Here we observe that some tokens are incorrectly split, particularly words marked as “marked” appear to exhibit unusual segmentation patterns, at least from the perspective of large models and negative prefixes.

# %%
import torch
from transformer_lens import HookedTransformer

# EN: Initialize the HookedTransformer model with pre-trained gpt2-small weights.
# CN: 使用预训练的 gpt2-small 权重初始化 HookedTransformer 模型。
model = HookedTransformer.from_pretrained("gpt2-small")

# EN: Prepare contrastive word pairs to analyze the representation of negation.
# CN: 准备对比词对，用于分析否定含义的表示。
word_pairs = [("happy", "unhappy"), ("clear", "unclear"), ("likely", "unlikely")]

# EN: Function to extract residual stream activations for calculating negation vectors.
# CN: 提取残差流激活值以计算否定向量的函数。
def get_negation_vector(model, pos=-1, layer=6):
    diffs = []
    for base, negated in word_pairs:
        # EN: Get activations for 'base' (e.g., happy).
        # CN: 获取 base 的激活值 (例如 happy)。
        # EN: Get activations for 'negated' (e.g., the 'un' token in 'unhappy').
        # CN: 获取 negated 的激活值 (例如 un-happy 中的 un)。
        # EN: Calculate the difference and average them to find the "Negation Direction" for this layer.
        # CN: 计算差值并平均，得到该层的 "Negation Direction" (否定方向)。
        pass
    return torch.stack(diffs).mean(0)

print("done")

# %%
# EN: Categorize words into different groups based on tokenization and semantic structure.
# CN: 根据分词结果和语义结构对单词进行实验分类。
def classify_words_for_experiment(word_list):
    groups = {
        "Group A (True Negation - Split)": [],
        "Group B (Pseudo - Atomic)": [],
        "Group C (Pseudo - Split)": [],
        "Discarded (Other)": []
    }

    # EN: Print the table header for the classification process.
    # CN: 打印分类过程的表格表头。
    print(f"{'Word':<15} | {'Tokens':<30} | {'Category'}")
    print("-" * 70)

    for word in word_list:
        # EN: Retrieve the list of string tokens from the model.
        # CN: 获取模型的 token 字符串列表。
        str_tokens = model.to_str_tokens(word)
        # EN: Note: GPT-2 tokens often include a leading space (e.g., ' un', 'happy').
        # CN: 注意：GPT-2 的分词通常带有前导空格，比如 ' un', 'happy'。

        # EN: Core classification logic based on token length and prefix content.
        # CN: 基于 token 长度和前缀内容的核心分类逻辑。
        if len(str_tokens) <= 2: # EN: Only [BOS, word] / CN: 只有 [起始符, 单词]
            category = "Group B (Pseudo - Atomic)"
            groups[category].append(word)

        elif len(str_tokens) >= 3 and "un" in str_tokens[1].lower():
            # EN: The word is split, and the second token (first actual) contains 'un'.
            # CN: 单词被切分，且第二个 token（即第一个实际字符）包含 'un'。

            # EN: Heuristic: Determine if the remainder is a common word (True) or a root/suffix (Pseudo).
            # CN: 启发式规则：判断剩余部分是独立常用词（真否定）还是词根/后缀（伪否定）。

            # EN: Use a predefined list of known pseudo-negations for demonstration purposes.
            # CN: 出于演示目的，使用预定义的已知伪否定列表进行区分。
            known_pseudos =     ["uncle","underwear",
                  "unicorn",
                  "unit",
                  "units",
                  "universe",
                  "university",
                  "union",
                  "unity",
                  "uniform",
                  "underpass",
                  "underground",
                  "underworld",
                  "undertone",
                  "undertaker",
                  "undercurrent",
                  "underbrush",
                  "underclass",
                  "undergraduate",
                  "underling",
                  "underpants",
                  "understatement",
                  "understructure",
                  "undertow",

                  # Verbs (lexicalized, not negation)
                  "understand",
                  "understood",
                  "undertake",
                  "undertakes",
                  "undertaking",
                  "undergo",
                  "underwent",
                  "undergone",
                  "unite",
                  "united",
                  "unites",
                  "unifying",
                  "unload",
                  "unloaded",
                  "unloading",
                  "uncover",
                  "uncovered",
                  "uncovering",
                  "unleash",
                  "unleashed",
                  "unleashing",
                  "undermine",
                  "undermined",
                  "undermining",
                  "underpin",
                  "underpinned",
                  "underpinning",

                  # Adjectives (non-negating / lexical)
                  "unique",
                  "uniformed",
                  "universal",
                  "unilateral",
                  "underground",
                  "uncanny",
                  "untold",
                  "unwieldy",
                  "unison",
                  "unisex",
                  "untamed",   # descriptive, not logical negation
                  "untitled", # conventional label, not logical negation

                  # Measurement / math / science
                  "unitary",
                  "univariate",
                  "univariate",
                  "universe",
                  "units",
                  "unity",
                  "unification",
                  "unifier",
                  "uniformity",
                  "universality",

                  # Proper nouns / names / entities
                  "unix",
                  "unicode",
                  "unesco",
                  "unicef",
                  "unilever",
                  "uniswap",
                  "unrealengine",
                  "unityengine",
                  "unreal",
                  "unitedstates",
                  "unitedkingdom",
                  "universitycollege",
                  "unionbank",

                  # Places & institutions
                  "unionstation",
                  "universitytown",
                  "universitycampus",
                  "undergroundstation",

                  # Misc lexicalized forms
                  "underbelly",
                  "underfoot",
                  "underhand",
                  "undercover",
                  "undercut",
                  "underpay",
                  "undershirt",
                  "underside",
                  "undersigned",
                  "undertaker",
                  "undertaking",
              ]

            if word in known_pseudos:
                category = "Group C (Pseudo - Split)"
            else:
                category = "Group A (True Negation - Split)"

            groups[category].append(word)

        else:
            category = "Discarded (Other)"
            groups[category].append(word)

        print(f"{word:<20} | {str(str_tokens):<50} | {category}")

    return groups

# --- EN: Input candidate word library / CN: 输入你的候选词库 ---
candidates = [
    # EN: True Negations (Expected in Group A)
    # CN: 真正语义上的否定词 (预期归入 Group A)
    "unable",
    "unacceptable",
    "unaccountable",
    "unaware",
    "unbelievable",
    "unbiased",
    "unclear",
    "uncomfortable",
    "uncommon",
    "unconscious",
    "undecided",
    "undefined",
    "undesirable",
    "unequal",
    "unethical",
    "unfair",
    "unfamiliar",
    "unfortunate",
    "unhappy",
    "unhealthy",
    "unimportant",
    "unintelligent",
    "unintentional",
    "uninteresting",
    "unjustified",
    "unlawful",
    "unlikely",
    "unnecessary",
    "unofficial",
    "unorganized",
    "unpleasant",
    "unpopular",
    "unpredictable",
    "unprofessional",
    "unqualified",
    "unreasonable",
    "unsafe",
    "unsatisfied",
    "unsecure",
    "unsuccessful",
    "unsuitable",
    "unsure",
    "unusual",
    "unwilling",
    "unacknowledged",
    "unconvinced",
    "uncertain",
    "uninterested",
    "unknown",
    "unambiguous",
    "unbalanced",
    "unbounded",
    "unclassified",
    "uncorrelated",
    "undecidable",
    "undetermined",
    "uninitialized",
    "unlabelled",
    "unobservable",
    "unverified",
    "unmotivated",
    "unprepared",
    "unrewarding",
    "unskilled",
    "unstated",
    "unsupported",

    # EN: Pseudo Negations (Expected in Group B or C)
    # CN: 形式上的否定词/伪否定 (预期归入 Group B 或 C)
    # Core nouns
    "uncle",
    "underwear",
    "unicorn",
    "unit",
    "units",
    "universe",
    "university",
    "union",
    "unity",
    "uniform",
    "underpass",
    "underground",
    "underworld",
    "undertone",
    "undertaker",
    "undercurrent",
    "underbrush",
    "underclass",
    "undergraduate",
    "underling",
    "underpants",
    "understatement",
    "understructure",
    "undertow",

    # Verbs (lexicalized, not negation)
    "understand",
    "understood",
    "undertake",
    "undertakes",
    "undertaking",
    "undergo",
    "underwent",
    "undergone",
    "unite",
    "united",
    "unites",
    "unifying",
    "unload",
    "unloaded",
    "unloading",
    "uncover",
    "uncovered",
    "uncovering",
    "unleash",
    "unleashed",
    "unleashing",
    "undermine",
    "undermined",
    "undermining",
    "underpin",
    "underpinned",
    "underpinning",

    # Adjectives (non-negating / lexical)
    "unique",
    "uniformed",
    "universal",
    "unilateral",
    "underground",
    "uncanny",
    "untold",
    "unwieldy",
    "unison",
    "unisex",
    "untamed",   # descriptive, not logical negation
    "untitled", # conventional label, not logical negation

    # Measurement / math / science
    "unitary",
    "univariate",
    "univariate",
    "universe",
    "units",
    "unity",
    "unification",
    "unifier",
    "uniformity",
    "universality",

    # Proper nouns / names / entities
    "unix",
    "unicode",
    "unesco",
    "unicef",
    "unilever",
    "uniswap",
    "unrealengine",
    "unityengine",
    "unreal",
    "unitedstates",
    "unitedkingdom",
    "universitycollege",
    "unionbank",

    # Places & institutions
    "unionstation",
    "universitytown",
    "universitycampus",
    "undergroundstation",

    # Misc lexicalized forms
    "underbelly",
    "underfoot",
    "underhand",
    "undercover",
    "undercut",
    "underpay",
    "undershirt",
    "underside",
    "undersigned",
    "undertaker",
    "undertaking",
]

# EN: Execute classification and store the structured experimental data.
# CN: 执行分类并存储结构化的实验数据。
experiment_data = classify_words_for_experiment(candidates)

# %% [markdown]
# We have constructed three distinct types of vocabulary, each with specific distinctions from the others.

# %%
def get_embedding_vector(word):
    # EN: Get the word embedding (vector before Layer 0 input).
    # This is a simple lookup table operation.
    # CN: 获取单词的 Embedding (Layer 0 输入前的向量)。
    # 这是一个简单的查找表操作。
    token_id = model.to_tokens(word, prepend_bos=False).item()
    return model.W_E[token_id]

# EN: Define base negation pairs manually to calculate the benchmark.
# Note: We compare the overall semantics of root and un-word, or use 'un' token specifically.
# For projection experiments, we calculate: Direction = Embedding(unhappy) - Embedding(happy).
# CN: 我们需要手动定义几对干净的 True Negation 用于计算基准。
# 注意：这里我们只取 Root 和 Un-Word 的整体语义对比，或者取 'un' 这个 token 的向量。
# 为了几何投影实验，我们通常计算：Direction = Embedding(unhappy) - Embedding(happy)。

# EN: Assumption: Directly use the embedding of the " un" token as the "negation direction" base.
# CN: 方案：直接使用 "un" 这个 Token 的 Embedding 向量作为“否定方向”的基准。
un_token_id = model.to_single_token(" un") # EN: Note the space / CN: 注意空格
negation_vector = model.W_E[un_token_id]

print(f"Negation Vector Shape: {negation_vector.shape}")

# %%
import torch
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
import numpy as np

# EN: Ensure the model is loaded in the current environment.
# CN: 确保模型已加载。
if 'model' not in locals():
    model = HookedTransformer.from_pretrained("gpt2-small")

# 1. EN: Define a rigorous contrastive dataset.
# 1. CN: 定义严谨的对比数据集。
pairs = [
    ("happy", "unhappy"),
    ("likely", "unlikely"),
    ("clear", "unclear"),
    ("pleasant", "unpleasant"),
    ("known", "unknown"),
    ("able", "unable"),

    ("aware", "unaware"),
    ("fair", "unfair"),
    ("safe", "unsafe"),
    ("sure", "unsure"),
    ("certain", "uncertain"),
    ("common", "uncommon"),
    ("usual", "unusual"),
    ("comfortable", "uncomfortable"),
    ("healthy", "unhealthy"),
    ("important", "unimportant"),
    ("necessary", "unnecessary"),
    ("acceptable", "unacceptable"),
    ("professional", "unprofessional"),
    ("successful", "unsuccessful"),
    ("popular", "unpopular"),
    ("willing", "unwilling"),
    ("reasonable", "unreasonable"),
    ("qualified", "unqualified"),
    ("predictable", "unpredictable"),
    ("reliable", "unreliable"),
    ("stable", "unstable"),
    ("balanced", "unbalanced"),
    ("prepared", "unprepared"),
    ("motivated", "unmotivated"),
    ("skilled", "unskilled"),
    ("certain", "uncertain"),
    ("interested", "uninterested"),
    ("biased", "unbiased"),
    ("ethical", "unethical"),
    ("lawful", "unlawful"),
    ("equal", "unequal"),
    ("fair-minded", "unfair-minded"),
]

def calculate_layerwise_negation_vector(model, pairs):
    num_layers = model.cfg.n_layers
    device = model.cfg.device # EN: Get model device (cuda or cpu) / CN: 获取模型所在的设备 (cuda 或 cpu)

    # EN: Initialize tensors on the correct device.
    # CN: 初始化在正确的设备上。
    negation_vectors = torch.zeros((num_layers, model.cfg.d_model), device=device)

    print(f"Calculating average negation vectors for {len(pairs)} word pairs on device: {device}...")

    for layer in range(num_layers):
        layer_diffs = []
        for pos_word, neg_word in pairs:
            # EN: Run inference and cache activations.
            # CN: 运行并缓存。
            _, cache_pos = model.run_with_cache(f" {pos_word}")
            _, cache_neg = model.run_with_cache(f" {neg_word}")

            hook_name = f"blocks.{layer}.hook_resid_post"

            # EN: Extract residual stream vectors at the last position.
            # CN: 提取最后一个位置的残差流向量。
            vec_pos = cache_pos[hook_name][0, -1, :]
            vec_neg = cache_neg[hook_name][0, -1, :]

            # EN: Compute the difference vector.
            # CN: 计算差值。
            layer_diffs.append(vec_neg - vec_pos)

        # EN: Calculate mean difference as the negation direction for this layer.
        # CN: 计算平均值作为该层的否定方向。
        negation_vectors[layer] = torch.stack(layer_diffs).mean(dim=0)

    return negation_vectors

# EN: Execute calculation.
# CN: 执行计算。
negation_directions = calculate_layerwise_negation_vector(model, pairs)
print(f"Calculation complete! Layers: {negation_directions.shape[0]}, Dimension: {negation_directions.shape[1]}")

# %%
import torch.nn.functional as F

def plot_projection_trajectories(model, negation_dirs, test_words):
    plt.figure(figsize=(10, 6))
    device = model.cfg.device

    for word in test_words:
        projections = []
        # EN: Run with cache in the correct context.
        # CN: 确保输入在正确的语境下运行。
        _, cache = model.run_with_cache(f"{word}")

        for layer in range(model.cfg.n_layers):
            word_vec = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]
            neg_vec = negation_dirs[layer] # EN: Already on the same device / CN: 已经在同一个 device 上了

            # EN: Calculate cosine similarity between word vector and negation direction.
            # CN: 计算单词向量与否定方向之间的余弦相似度。
            sim = F.cosine_similarity(word_vec, neg_vec, dim=0).item()
            projections.append(sim)

        plt.plot(projections, label=f"Word:{word}", marker='o', markersize=4)

    plt.title("Layer-wise Projection onto Negation Direction")
    plt.xlabel("Layer Index")
    plt.ylabel("Cosine Similarity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# --- EN: Re-run tests / CN: 重新运行测试 ---
test_candidates = [" bad", " apple", " unique", " unit", " university", " unfortunate", " understand"]
plot_projection_trajectories(model, negation_directions, test_candidates)

# %% [markdown]
# Here we can observe that while the negative projections of all words have decreased, if we examine the green and purple lines, we find that the patterns of these two words are remarkably similar. Specifically, “unique” and “university” initially cluster near “bad,” and later on, the curves for ‘unique’ and “bad” converge closely.

# %%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_layer_pca(model, layer, target_words):
    # 1. EN: Collect vectors for all words at the specified layer.
    # CN: 1. 收集所有词在该层的向量。
    vectors = []
    labels = []

    for word in target_words:
        _, cache = model.run_with_cache(f" {word}")
        # EN: Extract vector of the specified layer (Last token).
        # CN: 提取指定层的向量 (Last token)。
        vec = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :].cpu().detach().numpy()
        vectors.append(vec)
        labels.append(word)

    # 2. EN: Perform PCA to reduce dimensionality to 2D.
    # CN: 2. PCA 降维到 2D。
    pca = PCA(n_components=2)
    reduced_vecs = pca.fit_transform(vectors)

    # 3. EN: Plotting.
    # CN: 3. 绘图。
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_vecs[:, 0], reduced_vecs[:, 1])

    for i, label in enumerate(labels):
        plt.annotate(label, (reduced_vecs[i, 0], reduced_vecs[i, 1]), fontsize=12)

    plt.title(f"PCA Visualization of Word Embeddings at Layer {layer}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.show()

# EN: Word list: mixing true negations, pseudo-negations, and unrelated words.
# CN: 单词表：混合真否定、伪否定、无关词。
words_to_plot = [
    "happy", "good", "excellent",       # EN: Positive / CN: 正面
    "unhappy", "bad", "terrible",       # EN: Negative / CN: 负面
    "unclear", "unlikely", "undo",      # EN: True Negation / CN: 真否定
    "unique", "university", "union",    # EN: Pseudo Negation (Suspects) / CN: 伪否定 (嫌疑人)
    "apple", "cat", "table"             # EN: Neutral/Unrelated / CN: 中性/无关
]

# EN: Compare Layer 4 (Bias Layer) and Layer 10 (Correction Layer).
# CN: 对比 Layer 4 (偏见层) 和 Layer 10 (修正层)。
plot_layer_pca(model, layer=4, target_words=words_to_plot)
plot_layer_pca(model, layer=10, target_words=words_to_plot)

# %%
!pip install seaborn

# %%
import pandas as pd

# %%
true_neg_list = [
    "unhappy", "unlikely", "unclear", "unable", "unknown",
    "unsafe", "unpleasant", "uncertain", "unusual", "unaware",
    "uncomfortable", "unnecessary", "unsuccessful", "unimportant",
    "unwilling", "unafraid", "unclean", "unfair", "unlucky", "unreal"
]

# EN: Mixed pseudo-negation words containing "Atomic" (unique) and "Split" (university).
# Intended to demonstrate the ubiquity of bias.
# CN: 混合了 "Atomic" (unique) 和 "Split" (university) 的伪否定词。
# 旨在展示偏见的普遍性。
pseudo_list = [
    "unique", "unit", "union", "university", "universe",
    "understand", "uncle", "uniform", "unify", "unison",
    "under", "universal", "unicorn", "unity", "unite",
    "undergo", "undertake", "undermine", "units", "unions"
]

random_list = [
    "apple", "table", "car", "house", "run",
    "jump", "blue", "red", "computer", "sky",
    "water", "book", "music", "time", "person",
    "year", "way", "day", "thing", "man"
]

# %%
def get_layer_activations(model, words, layer):
    """
    EN: Retrieve Residual Stream vectors for a set of words at a specified layer.
    CN: 获取一组词在指定层的 Residual Stream 向量。
    """
    vectors = []
    for word in words:
        # EN: Add space prefix to match Tokenizer convention.
        # CN: 添加空格前缀以匹配 Tokenizer 习惯。
        text = f" {word}"
        _, cache = model.run_with_cache(text)
        # EN: Get [Batch, Pos, Dim] -> [0, -1, :] (Last token).
        # CN: 获取 [Batch, Pos, Dim] -> [0, -1, :] (最后一个 token)。
        vec = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]
        vectors.append(vec)
    return torch.stack(vectors)

def calculate_negation_vector(model, layer):
    """
    EN: Calculate a rigorous negation direction (Unhappy - Happy).
    CN: 计算严谨的否定方向 (Unhappy - Happy)。
    """
    pairs = [("happy", "unhappy"), ("clear", "unclear"),
             ("likely", "unlikely"), ("able", "unable")]
    diffs = []
    for pos, neg in pairs:
        v_pos = get_layer_activations(model, [pos], layer)[0]
        v_neg = get_layer_activations(model, [neg], layer)[0]
        diffs.append(v_neg - v_pos)
    return torch.stack(diffs).mean(dim=0)

def compute_scores(word_list, ref_vec):
    vecs = get_layer_activations(model, word_list, target_layer)
    # EN: Calculate Cosine Similarity.
    # CN: 计算 Cosine Similarity。
    scores = F.cosine_similarity(vecs, ref_vec.unsqueeze(0), dim=1)
    return scores.detach().cpu().numpy().tolist()

# %%
target_layer = 4
print(f"Processing Layer {target_layer}...")
neg_vec = calculate_negation_vector(model, target_layer)

true_neg_scores = compute_scores(true_neg_list, neg_vec)
pseudo_scores = compute_scores(pseudo_list, neg_vec)
random_scores = compute_scores(random_list, neg_vec)

# %%
import seaborn as sns

data = {
    "Cosine Similarity": true_neg_scores + pseudo_scores + random_scores,
    "Group": ["True Negation"] * len(true_neg_scores) +
             ["Pseudo Negation"] * len(pseudo_scores) +
             ["Random Control"] * len(random_scores)
}
df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))

custom_palette = {"True Negation": "#4c72b0", "Pseudo Negation": "#c44e52", "Random Control": "#dddddd"}

# EN: Draw boxplot.
# CN: 画箱线图。
sns.boxplot(x="Group", y="Cosine Similarity", data=df, palette=custom_palette, width=0.5)
# EN: Draw stripplot to show individual data points.
# CN: 画散点图 (展示具体数据点)。
sns.stripplot(x="Group", y="Cosine Similarity", data=df, color=".25", alpha=0.6, jitter=True)

plt.title(f"Statistical Evidence of Prefix Bias at Layer {target_layer}", fontsize=14)
plt.ylabel("Projection on Negation Vector (Higher = More Negative)", fontsize=12)
plt.axhline(0, color='black', linestyle='--', alpha=0.3) # EN: Add zero line / CN: 添加 0 线
plt.grid(axis='y', alpha=0.3)

plt.show()

# %%
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from typing import List

# %%
train_pairs = [
    ("happy", "unhappy"),
    ("clear", "unclear"),
    ("likely", "unlikely"),
    ("common", "uncommon"),
    ("aware", "unaware"),
    ("pleasant", "unpleasant"),
    ("healthy", "unhealthy"),
    ("safe", "unsafe"),
    ("fair", "unfair"),
    ("kind", "unkind"),
    ("lucky", "unlucky"),
    ("real", "unreal"),
    ("usual", "unusual"),
    ("necessary", "unnecessary"),
    ("comfortable", "uncomfortable")
]

test_words = [
    "unique",
    "universe",
    "university",
    "union",
    "unit",
    "understand",  # 特殊关注：虽然它是 split 的，但语义不应是否定
    "uncle"
]

# %%
def get_layer_activations(words: List[str], layer: int, model) -> np.ndarray:
    """
    EN: Extract last token activations for each word in the list at a specified layer.
    CN: 提取单词列表中每个单词在指定层的 Last Token 激活值。
    """
    activations = []
    for word in words:
        # EN: Add space to simulate words within sentences (handling tokenizer differences).
        # CN: 添加空格以模拟句中单词 (处理 Tokenizer 差异)。
        text = f" {word}"
        # EN: Run model and cache results.
        # CN: 运行模型并缓存。
        _, cache = model.run_with_cache(text, names_filter=lambda x: x.endswith(f"hook_resid_post"))

        # EN: Extract residual stream: [batch, seq_len, d_model].
        # We take batch=0, seq_len=-1 (Last token).
        # CN: 提取残差流: [batch, seq_len, d_model]。
        # 我们取 batch=0, seq_len=-1 (最后一个 token)。
        layer_act = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :].cpu().detach().numpy()
        activations.append(layer_act)

    return np.array(activations)

# %%
n_layers = model.cfg.n_layers
results = []

print(f"Starting linear probe scanning for {n_layers} layers...")

for layer in range(n_layers):
    # --- A. EN: Prepare training data / CN: A. 准备训练数据 ---
    pos_words = [p[0] for p in train_pairs]
    neg_words = [p[1] for p in train_pairs]

    X_pos = get_layer_activations(pos_words, layer, model)
    X_neg = get_layer_activations(neg_words, layer, model)

    X_train = np.concatenate([X_pos, X_neg])
    # EN: Labels: 0 = Positive/Root, 1 = Negative/Negated.
    # CN: 标签：0 = 正面/原词, 1 = 负面/否定词。
    y_train = np.concatenate([np.zeros(len(X_pos)), np.ones(len(X_neg))])

    # --- B. EN: Train probe (Logistic Regression) / CN: B. 训练探针 (Logistic Regression) ---
    # EN: Set max_iter higher to ensure convergence.
    # CN: max_iter 设置高一些以保证收敛。
    clf = LogisticRegression(random_state=42, max_iter=2000, C=1.0)
    clf.fit(X_train, y_train)

    # EN: Record training accuracy to verify probe validity.
    # CN: 记录训练准确率 (验证探针是否有效)。
    train_acc = clf.score(X_train, y_train)

    # --- C. EN: Test Group C (Trap Words) / CN: C. 测试 Group C (陷阱词) ---
    X_test = get_layer_activations(test_words, layer, model)
    # EN: Get probability of belonging to "Class 1 (Negative)".
    # CN: 获取属于 "Class 1 (Negative)" 的概率。
    probs = clf.predict_proba(X_test)[:, 1]

    # --- D. EN: Record results / CN: D. 记录结果 ---
    for word, prob in zip(test_words, probs):
        results.append({
            "Layer": layer,
            "Word": word,
            "Prob_Negative": prob,
            "Train_Acc": train_acc
        })

# %%
df_results = pd.DataFrame(results)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))

# EN: Plot the line chart for probability trajectories.
# CN: 绘制概率轨迹折线图。
sns.lineplot(
    data=df_results,
    x="Layer",
    y="Prob_Negative",
    hue="Word",
    marker="o",
    linewidth=2.5,
    palette="viridis"
)

# EN: Add reference lines for the decision boundary and high confidence zone.
# CN: 添加决策边界和高置信度区域的参考线。
plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label="Decision Boundary (Uncertain)")
plt.axhspan(0.8, 1.0, color='red', alpha=0.1, label="High Confidence Negative")

plt.title("Layer-wise Probability of Pseudo-Negations being Classified as 'Negative'", fontsize=14, fontweight='bold')
plt.xlabel("Layer Depth (0 = Embedding, 11 = Final)", fontsize=12)
plt.ylabel("Probability of Negative Class (P_neg)", fontsize=12)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.ylim(-0.05, 1.05)
plt.tight_layout()

plt.show()

# EN: Print a brief textual report of the key findings.
# CN: 打印关键发现的简要文字报告。
print("\n--- Key Layer Results Summary ---")
high_bias_words = df_results[(df_results["Layer"].isin([0, 1, 2])) & (df_results["Prob_Negative"] > 0.7)]["Word"].unique()
print(f"Words with obvious bias in early layers (L0-L2): {list(high_bias_words)}")

resolved_words = df_results[(df_results["Layer"] == n_layers-1) & (df_results["Prob_Negative"] < 0.3)]["Word"].unique()
print(f"Words successfully corrected in the final layer (L{n_layers-1}): {list(resolved_words)}")

# %%
# Method 3 Extension: Dual-Probe Analysis
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

# EN: Prepare word pairs for morphological (structural) and semantic (meaning) analysis.
# CN: 准备用于形态（结构）和语义（含义）分析的词对。
morph_pairs = [
    ("happy", "unhappy"), ("clear", "unclear"), ("likely", "unlikely"),
    ("common", "uncommon"), ("aware", "unaware"), ("pleasant", "unpleasant"),
    ("healthy", "unhealthy"), ("safe", "unsafe"), ("kind", "unkind"),
    ("lucky", "unlucky"), ("real", "unreal"), ("true", "untrue")
]

sem_pairs = [
    ("good", "bad"),
    ("success", "failure"),
    ("love", "hate"),
    ("rich", "poor"),
    ("win", "lose"),
    ("brave", "coward"),
    ("friend", "enemy"),
    ("joy", "pain"),
    ("strong", "weak"),
    ("warm", "cold")
]

target_words = ["unique", "university", "understand"]

def get_layer_activations(words, layer, model):
    # EN: Helper function to extract activations for a list of words.
    # CN: 辅助函数，用于提取一系列单词的激活值。
    activations = []
    for word in words:
        text = f" {word}"
        _, cache = model.run_with_cache(text, names_filter=lambda x: x.endswith(f"hook_resid_post"))
        layer_act = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :].cpu().detach().numpy()
        activations.append(layer_act)
    return np.array(activations)

def train_and_predict(pairs, target_words, layer, model):
    # EN: Train a logistic regression probe on the given pairs and predict probabilities for target words.
    # CN: 在给定词对上训练逻辑回归探针，并预测目标词的概率。

    # EN: Extract training set
    # CN: 提取训练集
    pos_words = [p[0] for p in pairs]
    neg_words = [p[1] for p in pairs]

    X_pos = get_layer_activations(pos_words, layer, model)
    X_neg = get_layer_activations(neg_words, layer, model)

    X_train = np.concatenate([X_pos, X_neg])
    y_train = np.concatenate([np.zeros(len(X_pos)), np.ones(len(X_neg))]) # 0=Pos, 1=Neg

    # EN: Training
    # CN: 训练
    clf = LogisticRegression(random_state=42, max_iter=2000)
    clf.fit(X_train, y_train)

    # EN: Testing
    # CN: 测试
    X_test = get_layer_activations(target_words, layer, model)
    probs = clf.predict_proba(X_test)[:, 1] # EN: Probability of Negative / CN: 否定的概率
    return probs

results = []
n_layers = model.cfg.n_layers

print(f"Running Dual-Probe Analysis on {target_words}...")

for layer in range(n_layers):
    # Probe A: Morphological
    probs_morph = train_and_predict(morph_pairs, target_words, layer, model)

    # Probe B: Semantic
    probs_sem = train_and_predict(sem_pairs, target_words, layer, model)

    for i, word in enumerate(target_words):
        results.append({"Layer": layer, "Word": word, "Probe Type": "Morphological (un-)", "Prob_Negative": probs_morph[i]})
        results.append({"Layer": layer, "Word": word, "Probe Type": "Semantic (good/bad)", "Prob_Negative": probs_sem[i]})

# %%
df = pd.DataFrame(results)

# EN: Visualize the results using FacetGrid to compare different words and probe types.
# CN: 使用 FacetGrid 可视化结果，对比不同单词和探针类型。
g = sns.FacetGrid(df, col="Word", height=5, aspect=1.2)
g.map_dataframe(sns.lineplot, x="Layer", y="Prob_Negative", hue="Probe Type", marker="o", linewidth=2.5)
g.add_legend()

# EN: Add reference lines for each plot.
# CN: 为每个图表添加参考线。
for ax in g.axes.flat:
    ax.axhline(0.5, color='black', linestyle='--', alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Prob of Being 'Negative'")

plt.subplots_adjust(top=0.85)
g.fig.suptitle("Decoupling Bias: Does 'Unique' look like 'Unhappy' or 'Bad'?", fontsize=16)
plt.show()

# %% [markdown]
# The blue line represents morphological analysis, while the red line indicates semantic analysis. A higher blue line indicates that the spelling carries more negative connotations, whereas the red line reflects the inherent negative meaning of the word itself.

# %%
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformer_lens import HookedTransformer
from typing import List, Tuple

model.eval()

# EN: Group words to test how attention heads respond to different structural contexts.
# CN: 对单词进行分组，测试注意力头在不同结构语境下的响应。
groups = {
    "Group A: True Negation (Split)": [
        "unhappy", "unclear", "unfair", "unkind", "unsafe", "unreal", "untrue"
    ],
    "Group C: Pseudo Negation (Split)": [
        "university", "universe", "uniform", "unison", "unify"
    ],
    # EN: Control Group: words with similar split structures but different prefixes to eliminate general attention noise.
    # CN: 控制组：具有相似切分结构但前缀不同的词，用于排除“泛泛的前缀关注”干扰。
    "Control: Other Prefix (Split)": [
        "understand", "underestimate", "interaction", "international"
    ]
}

def get_attention_to_prefix(word_list, model):
    """
    EN: Calculate average attention scores from Root Token to Prefix Token across the word list.
    Returns: [n_layers, n_heads] heatmap matrix.
    CN: 计算一组单词中，Root Token 对 Prefix Token 的平均注意力分数。
    返回: [n_layers, n_heads] 的热力图矩阵。
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # EN: Accumulator
    # CN: 累加器
    total_attn = np.zeros((n_layers, n_heads))
    count = 0

    for word in word_list:
        # EN: Construct input with a space to simulate natural context.
        # CN: 构造输入，加空格以模拟自然语境。
        text = f"The word is {word}"
        tokens = model.to_str_tokens(text)

        # EN: Heuristic logic to find split points (usually prefix and root are last two tokens).
        # CN: 寻找切分点的启发式逻辑（通常最后两个 token 是前缀和词根）。
        try:
            root_idx = -1
            prefix_idx = -2

            # EN: Run model to get activation cache.
            # CN: 运行模型获取 Cache。
            _, cache = model.run_with_cache(text, remove_batch_dim=True)

            # EN: Extract attention patterns for all layers and heads.
            # CN: 提取所有层、所有头的 Attention Pattern。
            for layer in range(n_layers):
                # Pattern shape: [n_heads, seq_len, seq_len]
                pattern = cache[f"blocks.{layer}.attn.hook_pattern"]

                # EN: Extract scores where Head H at Root position attends to Prefix position.
                # CN: 提取: Head H 在 Root 位置看向 Prefix 位置的分数。
                attn_scores = pattern[:, root_idx, prefix_idx].cpu().detach().numpy()

                total_attn[layer] += attn_scores

            count += 1

        except Exception as e:
            print(f"Skipping {word}: {e}")

    return total_attn / count if count > 0 else total_attn

results = {}

print("Starting panoramic attention scan...")
for group_name, words in groups.items():
    print(f"Analyzing {group_name}...")
    # EN: Filter words that cannot be split as expected for rigor.
    # CN: 过滤掉无法按照预期切分的词（为了严谨）。
    valid_words = []
    for w in words:
        toks = model.to_str_tokens(f" {w}")
        if len(toks) >= 2: # EN: Ensure at least two tokens / CN: 确保至少有两个 token
            valid_words.append(w)

    avg_attn = get_attention_to_prefix(valid_words, model)
    results[group_name] = avg_attn

fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

# EN: Standardize color scale range for comparison.
# CN: 统一色标范围，便于比较。
vmin, vmax = 0.0, 1.0

for i, (group_name, attn_map) in enumerate(results.items()):
    sns.heatmap(
        attn_map,
        ax=axes[i],
        cmap="Reds",
        vmin=vmin, vmax=vmax,
        cbar=(i==2) # EN: Show color bar only on the last plot / CN: 只在最后一张图显示色标
    )
    axes[i].set_title(group_name, fontsize=14)
    axes[i].set_xlabel("Head Index", fontsize=12)
    if i == 0:
        axes[i].set_ylabel("Layer Index", fontsize=12)

true_map = results["Group A: True Negation (Split)"]
pseudo_map = results["Group C: Pseudo Negation (Split)"]

# EN: Calculate difference map (True - Pseudo) to identify specific mechanism behaviors.
# CN: 计算差值图 (True - Pseudo) 以识别特定的机制行为。
diff_map = true_map - pseudo_map

# EN: Find heads with the largest difference (Disambiguation Heads).
# CN: 找出差异最大的头 (即：纠错头 Disambiguation Heads)。
disambiguation_heads = np.argwhere(diff_map > 0.3)

print("\n=== Mechanism Discovery ===")
print("Searching for 'Disambiguation Heads' (Heads attending to prefix in true negations but not in pseudo)...")

if len(disambiguation_heads) > 0:
    for layer, head in disambiguation_heads:
        print(f" -> Layer {layer}, Head {head}: Potential correction head (Difference score: {diff_map[layer, head]:.2f})")
else:
    print("No significant disambiguation heads found. Model may use the same mechanism for both (strong bias).")

# EN: Find 'Rigid Heads' that strongly attend to prefix regardless of semantics.
# CN: 找出“死板头” (Rigid Heads): 在两组中都强烈关注前缀的头。
rigid_mask = (true_map > 0.5) & (pseudo_map > 0.5)
rigid_heads = np.argwhere(rigid_mask)

print("\nSearching for 'Rigid Heads' (Heads that fixate on 'un-' regardless of semantics, likely bias sources)...")
for layer, head in rigid_heads:
    print(f" -> Layer {layer}, Head {head}: Potential bias injector (Avg Attn: {true_map[layer, head]:.2f})")

plt.tight_layout()
plt.show()

# %% [markdown]
# X-axis (Horizontal): Attention Heads (0 - 11). Represents different attention heads within the same layer.
# 
# Y-axis (Vertical): Layers (0 - 11). Represents the model's depth, from shallow layers (L0, near the input) to deep layers (L11, near the output).
# 
# Color: Deeper red (or higher numerical values) indicates “stronger attention.”
# 
# Specific meaning: When the model reads the latter part of a word (e.g., “happy” or ‘iversity’), this shows how much its Query vector is “fixated” on the preceding prefix (un-).

# %%
def get_layerwise_negation_dirs(model):
    """
    EN: Compute the (unhappy - happy) direction vector for every layer.
    CN: 计算每一层的 (unhappy - happy) 方向向量。
    """
    pos_text = " happy"
    neg_text = " unhappy"

    _, cache_pos = model.run_with_cache(pos_text, names_filter=lambda x: x.endswith("hook_resid_post"))
    _, cache_neg = model.run_with_cache(neg_text, names_filter=lambda x: x.endswith("hook_resid_post"))

    directions = {}
    n_layers = model.cfg.n_layers

    for layer in range(n_layers):
        # EN: [batch, seq, d_model] -> Take last token.
        # CN: [batch, seq, d_model] -> 取最后一个 token。
        v_pos = cache_pos[f"blocks.{layer}.hook_resid_post"][0, -1, :]
        v_neg = cache_neg[f"blocks.{layer}.hook_resid_post"][0, -1, :]

        # EN: Difference is the negation direction; normalize for consistency.
        # CN: 差值即为否定方向；进行归一化。
        diff = v_neg - v_pos
        diff = diff / diff.norm()
        directions[layer] = diff

    return directions

def analyze_attribution(target_word, model, negation_dirs):
    # EN: Analyze the contribution of Attention and MLP components to the negation direction.
    # CN: 分析 Attention 和 MLP 组件对否定方向的贡献。
    text = f" {target_word}"

    # EN: Explicitly cache 'hook_result' and 'hook_mlp_out' for attribution.
    # CN: 显式缓存 'hook_result' 和 'hook_mlp_out' 用于归因。
    def cache_filter(name):
        return name.endswith("hook_result") or name.endswith("hook_mlp_out")

    _, cache = model.run_with_cache(text, names_filter=cache_filter)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    head_contribs = np.zeros((n_layers, n_heads))
    mlp_contribs = np.zeros(n_layers)

    for layer in range(n_layers):
        dir_vec = negation_dirs[layer] # Tensor

        # EN: A. Analyze Attention Heads.
        # CN: A. 分析 Attention Heads。
        if f"blocks.{layer}.attn.hook_result" in cache:
            attn_out = cache[f"blocks.{layer}.attn.hook_result"][0, -1, :, :]

            # EN: Compute dot product between Head output and negation direction.
            # CN: 计算每个 Head 输出与否定方向的点积。
            scores = torch.matmul(attn_out, dir_vec).cpu().detach().numpy()
            head_contribs[layer] = scores
        else:
            print(f"Warning: Layer {layer} hook_result missing!")

        # EN: B. Analyze MLP components.
        # CN: B. 分析 MLP。
        if f"blocks.{layer}.hook_mlp_out" in cache:
            mlp_out = cache[f"blocks.{layer}.hook_mlp_out"][0, -1, :]
            mlp_score = torch.dot(mlp_out, dir_vec).cpu().detach().numpy()
            mlp_contribs[layer] = mlp_score

    return head_contribs, mlp_contribs

# %%
model.set_use_attn_result(True)

print("Calculating negation directions for each layer...")
neg_dirs = get_layerwise_negation_dirs(model)

words_to_compare = ["unhappy", "university"]
results = {}

for word in words_to_compare:
    print(f"Analyzing attribution for '{word}'...")
    results[word] = analyze_attribution(word, model, neg_dirs)

fig, axes = plt.subplots(2, 2, figsize=(18, 12), gridspec_kw={'height_ratios': [3, 1]})

# EN: Standardize color scale centered at 0.
# CN: 统一色标，以 0 为中心。
vmin, vmax = -2.0, 2.0

for i, word in enumerate(words_to_compare):
    head_data, mlp_data = results[word]

    # EN: Upper part: Heatmap for Attention Head contributions.
    # Red indicates positive (negation), Blue indicates negative (neutral/affirmative).
    # CN: 上半部分：Attention Heads 热力图。
    # 红色代表正向（否定），蓝色代表负向（中性/肯定）。
    sns.heatmap(
        head_data,
        ax=axes[0, i],
        cmap="RdBu_r",
        center=0,
        vmin=vmin, vmax=vmax,
        cbar=(i==1)
    )
    axes[0, i].set_title(f"Attention Head Contribution: '{word}'", fontsize=14)
    axes[0, i].set_xlabel("Head Index")
    axes[0, i].set_ylabel("Layer Index")

    # EN: Lower part: Bar chart for MLP contributions.
    # MLP contributions are often larger, so the scale is expanded.
    # CN: 下半部分：MLP 贡献条形图。
    # MLP 贡献通常较大，因此量程进行了放大。
    layers = np.arange(len(mlp_data))
    colors = ['red' if x > 0 else 'blue' for x in mlp_data]
    axes[1, i].bar(layers, mlp_data, color=colors)
    axes[1, i].set_title(f"MLP Contribution: '{word}'", fontsize=14)
    axes[1, i].set_xlabel("Layer")
    axes[1, i].set_ylabel("Contribution to Negation")
    axes[1, i].set_ylim(vmin*2, vmax*2)

plt.tight_layout()
plt.show()

print("\n=== Visualization Guide ===")
print("1. Red blocks/bars = Increased negation (Bias Injection)")
print("2. Blue blocks/bars = Decreased negation (Disambiguation/Restoration)")

# %%
def get_semantic_negation_dirs(model):
    """
    EN: Extract pure semantic directions using word pairs without 'un-' prefix to exclude spelling interference.
    Average multiple pairs like good/bad for robustness.
    CN: 使用不带 'un-' 的纯义词对提取纯语义方向，彻底排除拼写干扰。
    为了稳健性，使用多对词（如 good/bad）取平均值。
    """
    pairs = [
        (" good", " bad"),
        (" success", " failure"),
        (" win", " lose")
    ]

    n_layers = model.cfg.n_layers
    avg_diffs = {i: torch.zeros(model.cfg.d_model).to(model.cfg.device) for i in range(n_layers)}

    print("Extracting pure semantic directions (e.g., bad - good)...")

    for pos_word, neg_word in pairs:
        _, cache_pos = model.run_with_cache(pos_word, names_filter=lambda x: x.endswith("hook_resid_post"))
        _, cache_neg = model.run_with_cache(neg_word, names_filter=lambda x: x.endswith("hook_resid_post"))

        for layer in range(n_layers):
            v_pos = cache_pos[f"blocks.{layer}.hook_resid_post"][0, -1, :]
            v_neg = cache_neg[f"blocks.{layer}.hook_resid_post"][0, -1, :]
            # EN: Accumulate the difference. / CN: 累加差值。
            avg_diffs[layer] += (v_neg - v_pos)

    # EN: Normalize the resulting vectors. / CN: 归一化结果向量。
    directions = {}
    for layer in range(n_layers):
        diff = avg_diffs[layer]
        directions[layer] = diff / diff.norm()

    return directions

# %%
sem_dirs = get_semantic_negation_dirs(model)

words_to_compare = ["unhappy", "university"]
results_sem = {}

for word in words_to_compare:
    print(f"Analyzing semantic attribution for '{word}'...")
    results_sem[word] = analyze_attribution(word, model, sem_dirs)

# %%
fig, axes = plt.subplots(2, 2, figsize=(18, 12), gridspec_kw={'height_ratios': [3, 1]})
# EN: Slightly smaller range as semantic vectors often have smaller magnitudes.
# CN: 稍微调小一点范围，因为语义向量通常模长较小。
vmin, vmax = -1.5, 1.5

for i, word in enumerate(words_to_compare):
    head_data, mlp_data = results_sem[word]

    sns.heatmap(
        head_data,
        ax=axes[0, i],
        cmap="RdBu_r",
        center=0,
        vmin=vmin, vmax=vmax,
        cbar=(i==1)
    )
    axes[0, i].set_title(f"Attention Head (Semantic Impact): '{word}'", fontsize=14)
    axes[0, i].set_xlabel("Head Index")
    axes[0, i].set_ylabel("Layer Index")

    layers = np.arange(len(mlp_data))
    colors = ['red' if x > 0 else 'blue' for x in mlp_data]
    axes[1, i].bar(layers, mlp_data, color=colors)
    axes[1, i].set_title(f"MLP (Semantic Impact): '{word}'", fontsize=14)
    axes[1, i].set_ylim(vmin*2, vmax*2)

plt.suptitle("Correction Hypothesis Check: Do components inject 'Badness' or just 'Spelling'?", fontsize=16)
plt.tight_layout()
plt.show()

print(“\n=== Image Interpretation Guide ===”)
print(“1. Red squares/bars = Increase negative meaning (Bias Injection)”)
print(“2. Blue squares/bars = Eliminate negative meaning (Disambiguation/Restoration)”)
print(“3. Key comparison: In the ‘university’ diagram, are there any deep blue Head or MLP layers countering the red ones?”)

# %%
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from transformer_lens import utils
from tqdm import tqdm

# %%
clean_prompt = "The design is unique. It is"
corrupted_prompt = "The design is unhappy. It is"

print(f"\n[Prompt Setup]")
print(f"Clean:     '{clean_prompt}'")
print(f"Corrupted: '{corrupted_prompt}'")

# EN: Automatically locate patch positions.
# CN: 自动查找 patch 位置。
clean_tokens = model.to_str_tokens(clean_prompt)
corrupted_tokens = model.to_str_tokens(corrupted_prompt)

try:
    # EN: Note: GPT-2 Tokenizer usually keeps a leading space before words.
    # CN: 注意：GPT-2 Tokenizer 通常在单词前保留空格。
    target_pos = clean_tokens.index(" unique")
except ValueError:
    try:
        target_pos = clean_tokens.index("unique")
    except ValueError:
        print(f"Error: Could not find 'unique' in Clean Prompt. Tokens: {clean_tokens}")
        raise

# EN: Locate the ' un' prefix in the Corrupted Prompt (corresponding to 'unique').
# CN: 寻找 ' un' 在 Corrupted Prompt 中的位置 (对应 unique 的前缀)。
try:
    source_pos = corrupted_tokens.index(" un")
except ValueError:
    # EN: Fallback to 'un' without space if necessary.
    # CN: 如果找不到带空格的，尝试不带空格的 'un'。
    source_pos = corrupted_tokens.index("un")

print(f"\n[Token Alignment]")
print(f"Target (Clean): Index {target_pos} -> '{clean_tokens[target_pos]}'")
print(f"Source (Dirty): Index {source_pos} -> '{corrupted_tokens[source_pos]}'")

# %%
clean_prompt = "The design is unique. It is"
corrupted_prompt = "The design is unhappy. It is"

print(f"\n[Token Debugging]")
clean_tokens = model.to_str_tokens(clean_prompt)
corrupted_tokens = model.to_str_tokens(corrupted_prompt)

print(f"Clean Tokens:     {clean_tokens}")
print(f"Corrupted Tokens: {corrupted_tokens}")

# Target
# EN: Target word to be patched is 'unique'.
# CN: 我们要 patch 的是 'unique'。
target_candidates = [" unique", "unique"]
target_pos = None

for cand in target_candidates:
    if cand in clean_tokens:
        target_pos = clean_tokens.index(cand)
        print(f"[Target Found] '{cand}' at Index {target_pos}")
        break

if target_pos is None:
    raise ValueError(f"Could not find 'unique' in Clean Prompt. Please check the token list above.")

# Source
# EN: Look for 'un' prefix; if unhappy isn't split, fallback to the full word 'unhappy'.
# In this case, injecting the full 'unhappy' semantics into 'unique' remains a valid experiment.
# CN: 优先找前缀 'un'，如果找不到（说明 unhappy 没被切分），就退而求其次找整个 'unhappy'。
# 这种情况下，我们将把整个 'unhappy' 的语义注入给 'unique'，实验依然有效。
source_candidates = [" un", "un", " unhappy", "unhappy"]
source_pos = None

for cand in source_candidates:
    if cand in corrupted_tokens:
        source_pos = corrupted_tokens.index(cand)
        print(f"[Source Found] '{cand}' at Index {source_pos}")
        break

if source_pos is None:
    raise ValueError(f"Could not find 'un' or 'unhappy' in Corrupted Prompt. Please check the token list above.")

print(f"\n[Patching Plan]")
print(f"We will inject activation from Corrupted Index {source_pos} ('{corrupted_tokens[source_pos]}')")
print(f"into Clean Index {target_pos} ('{clean_tokens[target_pos]}')")
if target_pos != source_pos:
    print("Note: Index mismatch, the code will handle mapping automatically.")

# %%
positive_token = " good"
negative_token = " bad"

pos_token_id = model.to_single_token(positive_token)
neg_token_id = model.to_single_token(negative_token)

def logits_diff_metric(logits):
    # EN: Take the prediction distribution for the last token (following "It is").
    # CN: 取最后一个 token ("It is" 后面) 的预测分布。
    last_token_logits = logits[0, -1, :]
    return last_token_logits[neg_token_id] - last_token_logits[pos_token_id]

# %%
_, corrupted_cache = model.run_with_cache(corrupted_prompt)

# EN: Run Clean (Get health baseline).
# CN: 运行 Clean (获取健康基准)。
clean_logits, _ = model.run_with_cache(clean_prompt)
clean_diff = logits_diff_metric(clean_logits).item()

print(f"\n[Baselines]")
print(f"Clean Logit Diff (Bad - Good): {clean_diff:.4f}")
print("A more negative value indicates higher confidence in 'Good'; a positive value indicates a shift towards 'Bad'.")

# %%
# EN: Clean Run to establish baseline performance.
# CN: 运行 Clean 模式以建立基线性能。
clean_logits, clean_cache = model.run_with_cache(clean_prompt)
clean_diff = logits_diff_metric(clean_logits).item()

# EN: Corrupted Run to establish the impact of biased tokens.
# CN: 运行 Corrupted 模式以确立偏差 token 的影响。
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_prompt)
corrupted_diff = logits_diff_metric(corrupted_logits).item()

print(f"\n[Baselines]")
print(f"Clean Logit Diff (Bad - Good): {clean_diff:.4f} (should be < 0)")
print(f"Corrupted Logit Diff (Bad - Good): {corrupted_diff:.4f} (should be > 0)")
print(f"Total Damage Potential: {corrupted_diff - clean_diff:.4f}")

# %%
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
head_patches = np.zeros((n_layers, n_heads))

print(f"\nScanning all {n_layers * n_heads} Attention Heads...")

for layer in tqdm(range(n_layers)):
    for head in range(n_heads):
        # EN: Define Hook function: only replace the output of a specific Head at a specific position.
        # We inject the activation from corrupted source_pos into clean target_pos.
        # CN: 定义 Hook 函数：只替换特定 Head 在特定位置的输出。
        # 我们把 Corrupted 在 source_pos 的激活，塞给 Clean 的 target_pos。
        def patch_specific_head(result, hook):
            # result shape: [batch, seq, n_heads, d_model]
            result[:, target_pos, head, :] = corrupted_cache[hook.name][:, source_pos, head, :]
            return result

        # EN: Execute Patch.
        # CN: 运行 Patch。
        patched_logits = model.run_with_hooks(
            clean_prompt,
            fwd_hooks=[(f"blocks.{layer}.attn.hook_result", patch_specific_head)]
        )

        # EN: Record results.
        # CN: 记录结果。
        head_patches[layer, head] = logits_diff_metric(patched_logits).item()

# %%
# EN: Re-initializing and re-running the head-wise patch logic.
# CN: 重新初始化并运行 head-wise patch 逻辑。
head_patches = np.zeros((n_layers, model.cfg.n_heads))
for layer in tqdm(range(n_layers)):
    for head in range(model.cfg.n_heads):
        def patch_specific_head(result, hook):
            # result: [batch, seq, n_heads, d_model]
            result[:, target_pos, head, :] = corrupted_cache[hook.name][:, target_pos, head, :]
            return result

        patched_logits = model.run_with_hooks(
            clean_prompt,
            fwd_hooks=[(f"blocks.{layer}.attn.hook_result", patch_specific_head)]
        )
        head_patches[layer, head] = logits_diff_metric(patched_logits).item()

# %%
# EN: Calculate the net impact by subtracting the clean baseline from the patched results.
# CN: 通过从补丁结果中减去干净基准来计算净影响。
head_impact = head_patches - clean_diff

plt.figure(figsize=(12, 7))
# EN: Generate a heatmap to visualize the causal contribution of each attention head.
# CN: 生成热力图以可视化每个注意力头的因果贡献。
sns.heatmap(
    head_impact,
    cmap="Reds",
    center=0,
    cbar_kws={'label': 'Increase in "Bad" Probability (Logits)'}
)
plt.title(f"Toxic Head Map: Injecting '{corrupted_tokens[source_pos]}' into '{clean_tokens[target_pos]}'", fontsize=14)
plt.xlabel("Head Index")
plt.ylabel("Layer Index")
plt.show()

# EN: Instructions for Interpreting the Toxic Head Map:
# 1. Color Intensity: Darker red squares represent "Toxic Heads." These specific heads
#    are the primary drivers of prefix bias, where the 'un-' prefix incorrectly
#    triggers a negative sentiment response.
# 2. X-Y Coordinates: The Y-axis represents the layer depth, and the X-axis represents
#    the head index within that layer. This allows you to pinpoint the exact location
#    of bias injection in the transformer architecture.
# 3. Scale (Logits): The color bar indicates the increase in the logit for the "Bad"
#    token. A positive value means the specific head is successfully transmitting
#    the "negation" signal from the prefix to the root word.
print("\n[GUIDE: Interpreting the Toxic Head Map]")
print("1. Color Intensity: Darker red squares represent 'Toxic Heads'—the primary drivers of prefix bias.")
print("2. Coordinates: The Y-axis shows the Layer, and the X-axis shows the Head Index (e.g., Layer 4, Head 7).")
print("3. Causal Impact: The darker the cell, the more 'damage' this single head does by injecting bias.")

# %%
flat_indices = np.argsort(head_impact.flatten())
# EN: Take the top 3 indices (most positive impact).
# CN: 取最大的 3 个 (影响最正向的)。
top_3_indices = flat_indices[-3:][::-1]

print("These Heads are responsible for generating prefix bias:")
for i, idx in enumerate(top_3_indices):
    l = idx // n_heads
    h = idx % n_heads
    score = head_impact[l, h]
    print(f"Rank {i+1}: Layer {l} Head {h} (Impact: +{score:.4f} logits)")

# %%
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from tqdm import tqdm

# 1. EN: Initialization (skip if already loaded).
# CN: 1. 初始化 (如已加载可跳过)。
if 'model' not in locals():
    model = HookedTransformer.from_pretrained("gpt2-small")
model.set_use_attn_result(True)

# 2. EN: Experimental setup and intelligent index alignment.
# CN: 2. 实验设置与对齐 (智能查找)。
clean_prompt = "The design is unique. It is"
corrupted_prompt = "The design is unhappy. It is"
clean_tokens = model.to_str_tokens(clean_prompt)
corrupted_tokens = model.to_str_tokens(corrupted_prompt)

# EN: Automatically align indices for target and source tokens by searching for sub-strings.
# CN: 通过搜索子字符串自动对齐目标和源 token 的索引。
t_pos = next(i for i, t in enumerate(clean_tokens) if "unique" in t)
s_pos = next(i for i, t in enumerate(corrupted_tokens) if "un" in t)

# EN: Metric: Logit Difference (Bad - Good).
# A positive shift indicates the model is being "fooled" into a negative sentiment.
# CN: 指标：Logit Difference (Bad - Good)。
# 正向偏移表示模型被“诱导”产生了负面情感。
pos_id, neg_id = model.to_single_token(" good"), model.to_single_token(" bad")
def get_diff(logits): return (logits[0, -1, neg_id] - logits[0, -1, pos_id]).item()

# EN: Run baselines to establish "clean" and "corrupted" reference points.
# CN: 运行基准测试以建立“干净”和“损坏”的参考点。
_, corrupted_cache = model.run_with_cache(corrupted_prompt)
clean_logits, _ = model.run_with_cache(clean_prompt)
base_diff = get_diff(clean_logits)

# EN: Data computation core for causal mediation analysis.
# We measure how much the Logit Diff changes when we patch specific components.
# CN: 因果中介分析的数据计算核心。
# 我们测量当补丁特定组件时，Logit Diff 发生了多少变化。
n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
layer_results = {"Attn": [], "MLP": []}
head_results = np.zeros((n_layers, n_heads))

print("Extracting causal evidence chain...")
for l in tqdm(range(n_layers)):
    # EN: Layer-wise analysis (Macro level).
    # Patches the entire Attention or MLP output for the current layer.
    # CN: 逐层分析 (宏观层级)。
    # 补丁当前层整个 Attention 或 MLP 的输出。
    for comp in ["attn_out", "mlp_out"]:
        def patch_layer(res, hook):
            res[:, t_pos, :] = corrupted_cache[hook.name][:, s_pos, :]
            return res
        l_logits = model.run_with_hooks(clean_prompt, fwd_hooks=[(f"blocks.{l}.hook_{comp}", patch_layer)])
        layer_results["Attn" if "attn" in comp else "MLP"].append(get_diff(l_logits))

    # EN: Head-wise analysis (Micro level).
    # Patches individual attention heads to find specific "Toxic Heads".
    # CN: 逐头分析 (微观层级)。
    # 补丁单个注意力头以寻找具体的“毒性头”。
    for h in range(n_heads):
        def patch_head(res, hook):
            res[:, t_pos, h, :] = corrupted_cache[hook.name][:, s_pos, h, :]
            return res
        h_logits = model.run_with_hooks(clean_prompt, fwd_hooks=[(f"blocks.{l}.attn.hook_result", patch_head)])
        head_results[l, h] = get_diff(h_logits)

# EN: Evidence Block 1: Layer-wise causal bottleneck graph.
# CN: 证据板块 1: 宏观层级瓶颈图 (Layer-wise)。
plt.figure(figsize=(10, 4))
plt.plot(range(n_layers), np.array(layer_results["Attn"]) - base_diff, label="Attention Impact", marker='o', color='red')
plt.plot(range(n_layers), np.array(layer_results["MLP"]) - base_diff, label="MLP Impact", marker='s', color='blue')
plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.title("Evidence Block A: The Layer-wise Causal Bottleneck", fontsize=12)
plt.xlabel("Layer Index"), plt.ylabel("Logit Shift (Towards 'Bad')")
plt.legend(), plt.grid(alpha=0.2)
plt.show()

# EN: Instructions for Evidence Block A:
# This plot shows which layers are most susceptible to prefix bias.
# A high positive value in 'Attention Impact' suggests that the Attention mechanism in that layer
# is actively moving the model's prediction toward 'bad' when it sees the 'un-' prefix.
print("\n[GUIDE: Evidence Block A]")
print("1. X-axis represents the model depth. Early layers show where the bias starts.")
print("2. Y-axis shows the 'Damage' (Logit Shift). Higher values mean stronger causal influence.")
print("3. Compare Red (Attn) vs Blue (MLP) to see which component drives the prefix bias.")

# EN: Evidence Block 2: Toxic head heatmap fingerprint.
# CN: 证据板块 2: 微观毒性头热力图 (Head-wise)。
plt.figure(figsize=(10, 5))
sns.heatmap(head_results - base_diff, cmap="Reds", center=0, cbar_kws={'label': 'Damage'})
plt.title("Evidence Block B: The 'Toxic Head' Fingerprint", fontsize=12)
plt.xlabel("Head Index"), plt.ylabel("Layer Index")
plt.show()

# EN: Instructions for Evidence Block B:
# This heatmap identifies the exact 'Toxic Heads'.
# Darker red cells pinpoint specific attention heads that, when patched with the 'un-' activation,
# significantly flip the model's understanding of 'unique' from positive to negative.
print("\n[GUIDE: Evidence Block B]")
print("1. Each cell corresponds to a specific Attention Head (Layer, Head_Index).")
print("2. Dark Red cells are 'Toxic Heads'—the primary sources of the morphological bias.")
print("3. If multiple heads are red, the bias is distributed; if only one is dark, it is a single-point failure.")

# EN: Evidence Block 3: The Flip Score demonstration.
# CN: 证据板块 3: 行为翻转展示 (The Flip Score)。
best_idx = np.argmax(head_results)
best_l, best_h = best_idx // n_heads, best_idx % n_heads
top_score = head_results[best_l, best_h]

plt.figure(figsize=(6, 5))
bars = plt.bar(['Original (Unique)', 'After Surgery (Top Head)'], [base_diff, top_score], color=['green', 'darkred'])
plt.axhline(0, color='black', linewidth=1)
plt.title(f"Evidence Block C: Sentiment Flip by Head L{best_l}H{best_h}", fontsize=12)
plt.ylabel("Logit Difference (Bad - Good)")
plt.show()

# EN: Instructions for Evidence Block C:
# This bar chart demonstrates the 'causal power' of a single head.
# It compares the original prediction (Green) against the prediction after surgically
# injecting bias into only the most influential head (Dark Red).
print("\n[GUIDE: Evidence Block C]")
print(f"1. The Green bar shows the 'Clean' prediction: unique = good (negative logit diff).")
print(f"2. The Dark Red bar shows the result after patching ONLY Layer {best_l} Head {best_h}.")
print(f"3. If the bar crosses above the 0 line, a single head is sufficient to flip the model's logic.")


