
# Deep LLM & Transformer Engineering: A Comprehensive Technical Reference

This technical reference provides an exhaustive, end-to-end architectural guide to the internal workings of the Transformer, model compression (quantization), parameter-efficient fine-tuning (PEFT/LoRA), runtime execution, and schema-constrained structured extraction.

---

## Table of Contents
1. [Internal Workings of the Transformer](#1-internal-workings-of-the-transformer)
2. [BERT vs. GPT Architecture Comparison](#2-bert-vs-gpt-architecture-comparison)
3. [Quantization Fundamentals & Advanced LLM Engineering](#3-quantization-fundamentals--advanced-llm-engineering)
4. [LLM Fine-Tuning, PEFT, and Distributed Training](#4-llm-fine-tuning-peft-and-distributed-training)
5. [Structured Outputs, Domain Adaptation, and Information Extraction](#5-structured-outputs-domain-adaptation-and-information-extraction)

---

## 1. Internal Workings of the Transformer

An intuitive, mathematical breakdown of the standard Transformer architecture as described in *"Attention Is All You Need"*.

### Phase 1: The Input (Converting Text to Contextual Embeddings)

```
[Raw Text] -> [Tokenization] -> [Embedding Lookup] -> [Add Positional Encoding] -> [Encoder Input (X)]
```

#### 1. Tokenization & Input Embeddings
Neural networks process numbers, not text. Raw input text undergoes two primary transformations:
*   **Tokenization:** The text is segmented into sub-word chunks (tokens) and mapped to corresponding integer IDs from a pre-defined vocabulary dictionary.
*   **Embedding Matrix Lookup:** Each integer ID is mapped to a dense, continuous, learned vector of dimension $d_{\text{model}}$ (e.g., $512$ or $4096$). This vector captures initial static semantic relationships.

> **Intuition:** The input embedding acts like a static dictionary lookup. The word `"bank"` retrieves a vector describing its general semantic concepts, but it does not yet contain information on whether it represents a *"river bank"* or a *"money bank"*.

#### 2. The Problem of Sequence
Unlike Recurrent Neural Networks (RNNs) which process tokens sequentially from left to right, the Transformer processes all tokens in a sequence simultaneously. 
*   **Bag of Words Flaw:** Without sequence tracking, the sequences `"The man bit the dog"` and `"The dog bit the man"` would appear mathematically identical to the attention mechanism.
*   **The Solution:** An artificial "timestamp" or "position index" must be injected into each token's vector representation before it enters the network.

#### 3. Positional Encoding
The architecture utilizes interleaved sine and cosine functions of varying frequencies to generate a unique positional vector for each index $pos$:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

Where:
*   $pos$ is the position of the token in the sequence.
*   $i$ is the dimension index.

> **Intuition:** This acts like a multi-dimensional barcode indicating a word's exact position, allowing the network to compute relative distances between any two tokens instantly.

#### 4. The Final Contextual Embedding
The raw input embedding and the positional encoding are combined element-wise before entering the first Encoder block:

$$X_i = \text{Embedding}_{\text{Input}}(\text{token}_i) + PE_i$$

The resulting input matrix $X$ represents both static meaning and absolute position.

---

### Phase 2: The Encoder

The Encoder enriches static word embeddings into deeply connected contextual representations through a stack of identical blocks (usually $N=6$ or more).

```
                      +-----------------------------+
                      |       Input Matrix X        |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Multi-Head Self-Attention  | <---+ (Residual Connection)
                      +-----------------------------+     |
                                     |                    |
                                     v                    |
                      +-----------------------------+     |
                      |          Add & Norm         | ----+
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      | Position-wise Feed-Forward  | <---+ (Residual Connection)
                      +-----------------------------+     |
                                     |                    |
                                     v                    |
                      +-----------------------------+     |
                      |          Add & Norm         | ----+
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |     Enriched Context (Z)    |
                      +-----------------------------+
```

#### 5. Self-Attention: Creating Queries ($Q$), Keys ($K$), and Values ($V$)
For each word, the input matrix $X$ is multiplied by three learned projection weight matrices $W^Q, W^K, W^V \in \mathbb{R}^{d_{\text{model}} \times d_k}$:

$$Q = XW^Q$$

$$K = XW^K$$

$$V = XW^V$$

*   **Query ($Q$) - The Search Bar:** What the word is currently looking for (e.g., a pronoun looking for its antecedent).
*   **Key ($K$) - The Document Title:** What characteristics the word possesses (e.g., a noun identifying as a singular object).
*   **Value ($V$) - The Content:** The actual underlying semantic information of the word that will be extracted once a match is made.

#### 6. Self-Attention: Dot Product Affinity
The affinity or relevance of word $A$ to word $B$ is calculated using the dot product of their respective Query and Key vectors:

$$\text{Score} = Q \cdot K^T$$

A high dot product indicates strong alignment between the search requirements of the query and the characteristics of the key.

#### 7. Scale and Softmax
The raw scores are scaled down by the square root of the key dimension ($d_k$) to mitigate optimization issues, then passed through a Softmax function:

$$\text{Attention Weights} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

*   **Why Scale?** High-dimensional dot products can grow extremely large. Large values push the Softmax function into flat regions with near-zero gradients, stalling the backpropagation process (vanishing gradients).
*   **Softmax:** Converts raw scores into a normalized probability distribution summing to $1.0$.

#### 8. Extracting the Contextual Value
The normalized weights are multiplied by the Value ($V$) matrix to produce the final contextualized representation matrix $Z$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

> **Intuition:** If the pronoun `"it"` pays $80\%$ attention to `"dog"` and $20\%$ to `"bone"`, the new representation of `"it"` becomes a weighted blend containing $80\%$ of `"dog"`'s semantic values and $20\%$ of `"bone"`'s.

#### 9. Multi-Head Attention
To extract multiple distinct context relationships simultaneously, the queries, keys, and values are projected into $h$ separate representation subspaces:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

> **Intuition:** One attention head can track syntactic relations (*"Who did what?"*), another can track pronoun references (*"it" = "dog"*), and a third can monitor emotional sentiment, avoiding a single dominant relationship from obscuring other subtle features.

#### 10. Add & Norm (Residual Connections and Layer Normalization)
Every sub-layer in the block is wrapped in a residual connection followed by Layer Normalization:

$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

*   **Residual Connection (Add):** Adds the original input $x$ back to the computed output. This prevents vanishing gradients during backpropagation in deep networks by providing a direct gradient path.
*   **Layer Normalization (Norm):** Standardizes activations across the channel dimension to have a mean of $0$ and a variance of $1$, which stabilizes training and increases convergence speed.

#### 11. Position-wise Feed-Forward Network (FFN)
Following attention processing, each token's vector is passed independently through an identical, fully connected feed-forward network:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

The FFN introduces non-linear decision boundaries through an activation function (typically ReLU or GeLU) and is usually $4\times$ wider in its hidden layer than the input/output dimension.

> **Intuition:** While self-attention acts as a collaborative step where tokens share global information, the FFN is a individual processing step where each token processes its newly gathered context independently.

---

### Phase 3: The Decoder (Autoregressive Generation)

The Decoder utilizes both local and global context to generate output sequences token-by-token.

```
[Decoder Inputs] -> [Masked Self-Attention] -> [Cross-Attention (Queries from Dec, Keys/Values from Enc)] -> [FFN] -> [Linear & Softmax]
```

#### 12. Masked Attention
To prevent the model from looking at future tokens during parallel training, a causal look-ahead mask is applied to the self-attention logits. It forces all connections to future positions to $-\infty$, which evaluates to $0$ after Softmax:

$$\text{MaskedAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

Where the mask $M$ is defined as:

$$M_{ij} = \begin{cases} 0 & \text{if } i \ge j \\ -\infty & \text{if } i < j \end{cases}$$

#### 13. Cross-Attention Routing
This block bridges the Encoder and the Decoder. It matches the Decoder's current state with the Encoder's global memory representation:

$$\text{CrossAttention} = \text{softmax}\left(\frac{Q_{\text{dec}} K_{\text{enc}}^T}{\sqrt{d_k}}\right)V_{\text{enc}}$$

*   **Query ($Q_{\text{dec}}$):** Comes from the previous layer of the Decoder (representing the translation generated so far).
*   **Key ($K_{\text{enc}}$) & Value ($V_{\text{enc}}$):** Come directly from the final output of the Encoder (representing the source text context).

> **Intuition:** The Decoder asks: *"I have generated 'The dog jumped', now I need a verb from the source text. Here is my Query."* The Encoder's Keys match this request with the source word `"barked"`, and the Decoder extracts its corresponding Value representation.

#### 14. Output Projection (Linear & Softmax)
The continuous vector from the final Decoder block is projected back to the vocabulary dimension via a linear projection layer and normalized into token probabilities:

$$P(\text{word}_i) = \text{softmax}(W_{\text{vocab}} \cdot h_i + b)$$

The index with the highest probability is typically sampled as the next token.

---

## 2. BERT vs. GPT Architecture Comparison

| Architectural Feature | BERT (Bidirectional Encoder) | GPT (Autoregressive Decoder) |
| :--- | :--- | :--- |
| **Primary Building Block** | Transformer Encoder Stack | Transformer Decoder Stack (No cross-attention) |
| **Attention Mechanism** | Fully Bidirectional (Unmasked) | Causal Unidirectional (Lower-triangular mask) |
| **Context Window Processing**| Processes entire sequence simultaneously | Processes prompt in parallel; generates sequentially |
| **Input Representation** | $\text{Token} + \text{Segment} + \text{Position Embeddings}$ | $\text{Token} + \text{Position Embeddings}$ |
| **Primary Training Objective** | Masked Language Model (MLM) & Next Sentence Prediction (NSP) | Causal Language Modeling (CLM - Next Token Prediction) |
| **Inference Mode** | Non-autoregressive (Feature extraction/classification) | Autoregressive generation (KV-Cached generation loops) |
| **Primary Use Cases** | Sentiment Analysis, NER, Question Answering | Text Generation, Translation, Creative Writing, Reasoning |

### Key Differences in Training Objectives

*   **BERT (MLM):** Randomly masks $15\%$ of input tokens during pre-training. The model must predict these masked tokens by utilizing both left-and-right context [1].
*   **GPT (CLM):** Maximizes the likelihood of the next token given all preceding tokens:
    $$L_{\text{CLM}} = - \sum_{i=1}^{n} \log P(u_i \mid u_{<i}; \Theta)$$

---

## 3. Quantization Fundamentals & Advanced LLM Engineering

Quantization is the process of mapping continuous high-precision numbers (such as FP32 or FP16) to discrete lower-precision representations (such as INT8 or INT4) to reduce model size and accelerate inference.

```
FP32 (32-bit):  [Sign] [Exponent (8-bit)] [Mantissa (23-bit)]
FP16 (16-bit):  [Sign] [Exponent (5-bit)] [Mantissa (10-bit)]
INT8 (8-bit):   [Sign] [Value (7-bit)]
```

### 1. Mathematical Mapping
Quantization translates floating-point numbers to integers using a scale factor $S$ and a zero-point $Z$:

$$W_{\text{quantized}} = \text{round}\left(\frac{W}{S}\right) + Z$$

$$W_{\text{dequantized}} = S \cdot (W_{\text{quantized}} - Z)$$

Where:
*   **Scale Factor ($S$):** A floating-point number that scales the dynamic range.
*   **Zero-Point ($Z$):** An integer that maps the real value $0.0$ to the quantized domain (important for asymmetric distributions).

---

### 2. Runtime Memory Footprint Estimation

To calculate the base memory footprint required to load a model's weights into VRAM:

$$\text{Weight Memory (GB)} = \frac{\text{Parameter Count (in Billions)} \times \text{Bit-Width per Parameter}}{8}$$

#### Runtime Overhead
Loading a model for active inference requires accounting for additional runtime allocations:
*   **KV Cache:** Stores past token representations to prevent quadratic recomputation.
*   **Activation Memory:** Temporary tensors generated during intermediate steps of the forward pass.
*   **Execution Library Overhead:** Memory used by CUDA kernels, deep learning libraries (e.g., PyTorch), and system buffers.

> **Engineering Heuristic:** A $20\%$ buffer ($1.2\times$ multiplier) is typically added to estimate the active runtime memory footprint:
> $$\text{Total Runtime Memory Estimate} \approx \text{Weight Memory} \times 1.2$$

#### Empirical Memory Comparisons
The table below illustrates the trade-offs across various model scales and formats:

| Model | Parameter Count | Precision | Bit-Width | Weight Size | Est. Active Runtime | Compatible Target Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen 2.5** | 1.5B | FP16 | 16 | ~3.0 GB | ~3.6 GB | Modern mobile phones, entry-level tablets |
| **Qwen 2.5** | 1.5B | INT8 | 8 | ~1.5 GB | ~1.8 GB | Low-resource edge systems, Raspberry Pi 5 |
| **Qwen 2.5** | 1.5B | INT4 | 4 | ~0.75 GB | ~0.9 GB | Microcontrollers, ultra-low power devices |
| **Llama 3** | 8B | FP16 | 16 | ~16.0 GB | ~19.2 GB | Workstation GPUs (e.g., RTX 4090, Mac Studio) |
| **Llama 3** | 8B | INT8 | 8 | ~8.0 GB | ~9.6 GB | High-end laptops (16GB RAM) |
| **Llama 3** | 8B | INT4 | 4 | ~4.0 GB | ~4.8 GB | Standard laptops (8GB RAM), modern smartphones |
| **Llama 3** | 70B | FP16 | 16 | ~140.0 GB | ~168.0 GB| Multi-GPU server nodes (e.g., 2x A100 80GB) |
| **Llama 3** | 70B | INT8 | 8 | ~70.0 GB | ~84.0 GB | Pro workstation (e.g., Mac Studio 128GB Unified) |
| **Llama 3** | 70B | INT4 | 4 | ~35.0 GB | ~42.0 GB | Enthusiast setups (e.g., 2x RTX 3090/4090 24GB) |

---

### 3. Deep Dive: GGUF vs. GPTQ vs. AWQ

```
                 [ Unquantized FP16 LLM ]
                  /         |          \
                 v          v           v
            [ GGUF ]     [ GPTQ ]     [ AWQ ]
             (CPU/        (GPU-Only    (High-Perf
             Mixed)        vLLM)       Serving)
```

#### GGUF (GGML Universal Format)
*   **Format:** Single-file binary format that serializes both the model's weights and metadata (e.g., tokenizers, hyperparameters) into one container [3, 4].
*   **Key Feature (Layer Offloading):** Enables splitting model layers between VRAM and system CPU RAM. If a model needs $10$ GB of memory but the GPU only has $8$ GB, GGUF can offload remaining layers to system memory [4].
*   **Target:** Local deployment, CPU-only devices, macOS Unified Memory architectures, and single-user systems (e.g., Ollama, llama.cpp) [4].

#### GPTQ (Generalized Post-Training Quantization)
*   **Methodology:** Uses a calibration dataset (typically ~128 representative paragraphs) to evaluate the impact of quantization [5]. It calculates an approximate inverse Hessian matrix to adjust remaining unquantized weights layer-by-layer, minimizing quantization error [5].
*   **Execution:** Weights are stored as 4-bit integers on disk. During the forward pass, weights are unpacked on-the-fly to FP16 inside GPU registers for calculation [5].
*   **Target:** Dedicated GPU cloud serving where high throughput is required (e.g., vLLM nodes) [5].

#### AWQ (Activation-Aware Weight Quantization)
*   **Methodology:** Based on the empirical finding that not all weights are of equal importance [6]. Roughly $1\%$ of weights (the "salient channels") correspond to high-magnitude activations [6]. Quantizing these channels causes severe performance drops [6].
*   **Key Feature:** AWQ determines these critical channels via a calibration dataset, leaving them in higher precision or applying scaling vectors to protect them before quantizing the rest [6].
*   **Target:** GPU production servers requiring high accuracy at low bitwidths [6].

---

### 4. Advanced System-Level Execution

#### CPU Inference Optimization (SIMD & Threading)
*   **SIMD (Single Instruction, Multiple Data):** Processes multiple data values in a single clock cycle. Modern runtimes utilize AVX2 (256-bit registers), AVX-512 (512-bit registers), or ARM Neon (128-bit) to execute vectorized Fused Multiply-Add (FMA) commands, speeding up matrix multiplication.
*   **Thread Contention:** Hyper-threading splits physical CPU cores into logical cores that share physical Floating-Point Units (FPUs) and cache layers. Because LLM matrix operations saturate FPUs, running multiple threads per physical core can cause resource contention, cache misses, and pipeline stalls.
    > **Golden Rule:** Configure runtime thread counts to match the number of **physical** CPU cores rather than logical threads, and use CPU pinning (`Thread Affinity`) to prevent core-switching overhead.

#### Memory Mapping (`mmap`) & OS Paging
*   **Traditional file read (`fread`):** Copies file contents from disk into a kernel-space buffer, and then copies it again into a user-space buffer, which doubles bandwidth requirements and causes slow startup times.
*   **`mmap()` System Call:** Maps a file's byte sequences directly into the process's virtual address space [7].
    *   Initially, zero bytes are loaded into RAM. The OS maps file offsets to virtual memory addresses.
    *   **Demand Paging:** When the CPU references a memory address for an unmapped weight, a hardware page fault is raised, prompting the OS to load that specific page from disk into RAM.
    *   **Page Pinning (`mlock`):** Keeps the mapped memory pages locked in physical RAM, preventing the OS from swapping them to disk under system memory pressure [8].

#### Attention Head Optimization & KV Cache Memory Calculations
To avoid calculating Key and Value vectors for all past tokens at every step of autoregressive generation, past $K$ and $V$ matrices are stored in a cache buffer (the KV Cache). This modifies computational complexity from quadratic $O(N^2)$ to linear $O(N)$ during generation.

*   **Multi-Head Attention (MHA):** $1$ Query Head to $1$ Key Head and $1$ Value Head. The KV cache size scales with the number of heads.
*   **Multi-Query Attention (MQA):** Multiple Query Heads share a single Key and Value head. This reduces the KV Cache footprint but can impact accuracy due to reduced representational capacity.
*   **Grouped-Query Attention (GQA):** Query heads are clustered into groups (e.g., $8$ queries share $1$ KV head), balancing memory footprint and model capacity [9].

```
Multi-Head (MHA)            Grouped-Query (GQA)           Multi-Query (MQA)
  Q Q Q Q Q Q                 Q Q Q Q Q Q                    Q Q Q Q Q Q
  | | | | | |                 \ \ / / \ \                     \ \ / / /
  v v v v v v                  v   v   v                       v     v
  K K K K K K                  K   K   K                       K     K
  V V V V V V                  V   V   V                       V     V
```

To calculate the absolute memory footprint of the KV Cache in FP16 precision:

$$\text{Memory}_{\text{kvCache}} = 2 \times L \times H_{\text{kv}} \times D \times C \times B$$

Where:
*   $L$ = Number of layers
*   $H_{\text{kv}}$ = Number of KV Heads
*   $D$ = Head dimension ($\text{Hidden Size} / \text{Query Heads}$)
*   $C$ = Context window length
*   $B$ = Precision bytes per parameter ($2$ for FP16)
*   The initial multiplier of $2$ accounts for both the Key and Value matrices.

##### Real-World Example: Qwen 2.5 1.5B
Given parameters: $L=28$, $H_{\text{kv}}=2$, $D=128$, $B=2$.
*   **At $4,000$ context length:**
    $$\text{Memory}_{\text{kvCache}} = 2 \times 28 \times 2 \times 128 \times 4000 \times 2 \approx 114.7 \text{ MB}$$
*   **At $128,000$ context length:**
    $$\text{Memory}_{\text{kvCache}} = 2 \times 28 \times 2 \times 128 \times 128000 \times 2 \approx 3.67 \text{ GB}$$
    *(At long context windows, the KV Cache can exceed the memory footprint of the model weights themselves.)*

---

## 4. LLM Fine-Tuning, PEFT, and Distributed Training

Adapting a pre-trained foundation model for specific tasks typically involves targeted training runs.

```
                  [ Pre-Training Phase (Trillions of tokens) ]
                                       |
                                       v
                              [ Foundation Model ]
                                       |
                     +-----------------+-----------------+
                     |                                   |
                     v                                   v
         [ Full Fine-Tuning (FFT) ]              [ PEFT / LoRA ]
         Updates 100% of weights.                Freezes base weights.
         Requires high memory/compute.           Updates small adapter layer.
```

### 1. Full Fine-Tuning (FFT) Memory Distribution

Using a standard AdamW optimizer in a $16\text{-bit}$ training setup, the static memory distribution per parameter $N$ is calculated as follows:
*   **Base Weights:** $2 \text{ bytes} \times N$ (stored in FP16/BF16)
*   **Gradients:** $2 \text{ bytes} \times N$ (stored in FP16/BF16)
*   **Optimizer States (AdamW):**
    *   FP32 Master Copy of Weights: $4 \text{ bytes} \times N$
    *   FP32 First Moment (Momentum): $4 \text{ bytes} \times N$
    *   FP32 Second Moment (Variance): $4 \text{ bytes} \times N$
*   **Total Static Footprint:** $16 \text{ bytes}$ per parameter.

$$\text{Static Training Memory (GB)} = N \times 16 \text{ bytes}$$

*(An 8B parameter model requires at least $128$ GB of static memory, excluding activation memory and CUDA context buffers.)*

---

### 2. Parameter-Efficient Fine-Tuning (PEFT) & LoRA

To reduce the memory required for updates, PEFT freezes the base parameters $\Theta_{\text{base}}$ and only updates a small subset of adapter parameters $\Theta_{\text{adapter}}$.

```
                     Input Vector (x)
                            |
               +------------+------------+
               |                         |
               v                         v
       [ Frozen Base ]          [ Down-Project A ]  (Gaussian initialization)
       [  Weights W0 ]                   | (dim: r)
               |                         v
               |                 [ Up-Project B ]    (Zero initialized)
               |                         |
               +------------+------------+
                            v
                      Output Activation (y)
```

#### Low-Rank Adaptation (LoRA)
LoRA decomposes weight updates ($\Delta W$) into low-rank matrices [10]:

$$W = W_0 + \Delta W = W_0 + B \cdot A$$

Where:
*   $W_0 \in \mathbb{R}^{d \times k}$ (Frozen base weight matrix)
*   $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ (Trainable adapter matrices)
*   $r \ll \min(d, k)$ (The low-rank bottleneck, typically $r \in \{8, 16, 32\}$)

#### Initialization Strategy
To prevent architectural shifts at step $0$:
*   Matrix $A$ is initialized using a random Gaussian distribution.
*   Matrix $B$ is initialized entirely to $0$.
*   This ensures that $\Delta W = B \cdot A = 0$ at the start of training, so the model behavior matches the unadapted state exactly before updates begin.

#### QLoRA (Quantized Low-Rank Adaptation)
QLoRA optimizes this further by keeping the base model in a quantized state during training [11]:
*   **NF4 Data Type:** A 4-bit NormalFloat format designed for normally distributed neural network weights [11].
*   **Double Quantization (DQ):** Quantizes the quantization scale factors themselves (e.g., from 32-bit to 8-bit), saving additional VRAM [11].
*   **Paged Optimizers:** Uses CUDA Unified Memory to automatically swap optimizer memory pages to system RAM during memory spikes, preventing Out-Of-Memory (OOM) errors [11].

---

### 3. Execution Pathways

#### LoRA Training Pathway
```
                      [ Input Batch: FP16 ]
                                |
               +----------------+----------------+
               |                                 |
               v                                 v
       [ Frozen Base Weights ]          [ Trainable Adapters ]
       (VRAM as FP16)                   (VRAM as FP16)
       (Math in FP16)                   (Math in FP16)
               |                                 |
               +----------------+----------------+
                                v
                     [ Combined Output Activation ]
```

#### QLoRA Training Pathway
```
                      [ Input Batch: FP16 ]
                                |
               +----------------+----------------+
               |                                 |
               v                                 v
       [ Frozen Base Weights ]          [ Trainable Adapters ]
       (VRAM as 4-bit NF4)              (VRAM as FP16)
               |                        (Math in FP16)
     * On-the-fly Dequant *                      |
       (unpacks 4-bit to FP16)                   |
               |                                 |
       (Math in FP16)                            |
               |                                 |
               +----------------+----------------+
                                v
                     [ Combined Output Activation ]
```

---

### 4. Distributed Training Architectures (ZeRO)

For models too large to fit on a single GPU, the Zero Redundancy Optimizer (ZeRO) shards states across multiple processors:
*   **ZeRO-1:** Partitions the optimizer states across the GPU cluster.
*   **ZeRO-2:** Partitions both the optimizer states and gradients across GPUs.
*   **ZeRO-3:** Partitions the optimizer states, gradients, and model parameters across the cluster, streaming layers on-demand over interconnects like NVLink during the forward/backward passes.

---

## 5. Structured Outputs, Domain Adaptation, and Information Extraction

When using language models for structured information extraction, ensuring output conformity is important for downstream application pipelines.

### 1. Logit-Level Constrained Decoding

A model generates a probability distribution over vocabulary $V$ by applying Softmax to raw output logits $L_t$:

$$P(x_t = v | C) = \frac{\exp\left(L_t^{(v)}\right)}{\sum_{u \in V} \exp\left(L_t^{(u)}\right)}$$

In unconstrained decoding, any token can be sampled. Under schema-constrained decoding (e.g., using libraries like *Outlines*), a validator updates a logit-level mask to block invalid transition states:

$$L_t^{\text{masked}(v)} = \begin{cases} L_t^{(v)} & \text{if } v \in V_{\text{valid}} \\ -\infty & \text{otherwise} \end{cases}$$

This forces the probability of invalid tokens to $0$ before sampling occurs, ensuring syntactical correctness.

```
[Raw Logits] ---> [Apply Masking Filter (-inf for invalid)] ---> [Softmax] ---> [Valid Token Output]
```

#### Finite State Machine (FSM) Constraints
Constrained decoding engines often convert target JSON schemas or regular expressions into an FSM $(\Sigma, S, s_0, \delta, F)$ where:
*   $\Sigma$ is the input character alphabet.
*   $S$ is the set of states.
*   $s_0$ is the start state.
*   $\delta$ is the transition function ($S \times \Sigma \rightarrow S$).
*   $F$ is the set of accepting states.

```
Regex: [0-9]{3} (Zip Code Matcher)

             Digit (0-9)           Digit (0-9)           Digit (0-9)
 ( State 0 ) ------------> ( State 1 ) ------------> ( State 2 ) ------------> (( State 3 ))
      |                         |                         |                         |
 Any non-digit             Any non-digit             Any non-digit             Any character
      |                         |                         |                         |
      v                         v                         v                         v
 [-inf mask]               [-inf mask]               [-inf mask]               [-inf mask]
```

---

### 2. Implementation Paradigms: Outlines vs. Instructor

Here is a comparison of logit-level masking (Outlines) and application-level retry logic (Instructor).

```python
# ==========================================
# 1. OUTLINES: Logit-Level Constraint (Local)
# ==========================================
import outlines
from pydantic import BaseModel, Field

class MaterialConsignment(BaseModel):
    consignment_id: str = Field(description="Format: CON followed by exactly 4 digits.")
    weight_kg: float = Field(description="Weight of the container in kilograms.")
    status: str = Field(description="Allowed: 'In_Transit' or 'Arrived'.")

# Initialize the constrained model engine
model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")
generator = outlines.generate.json(model, MaterialConsignment)

document_text = """
Logistics Manifest Update:
Consignment unit CON-4911 was loaded at Rotterdam Terminal. Net weight registered at 1420.50 kg.
The current status of this container is In_Transit to Frankfurt.
"""

# The output is guaranteed to parse correctly as a MaterialConsignment object
structured_data = generator(document_text)
print(structured_data)


# ==========================================
# 2. INSTRUCTOR: Application-Level Retry (API)
# ==========================================
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
import datetime

client = instructor.from_openai(OpenAI())

class SupplierAuditLog(BaseModel):
    supplier_id: str = Field(..., description="Unique supplier code.")
    audit_date: str = Field(..., description="ISO 8601 date string.")
    compliance_rating: int = Field(..., description="Score from 1 to 100.")

    @field_validator('audit_date')
    @classmethod
    def validate_date_not_future(cls, value: str) -> str:
        parsed_date = datetime.datetime.strptime(value, "%Y-%m-%d").date()
        if parsed_date > datetime.date.today():
            raise ValueError(f"Audit date '{value}' cannot be in the future.")
        return value

    @field_validator('compliance_rating')
    @classmethod
    def validate_score_range(cls, value: int) -> int:
        if not (1 <= value <= 100):
            raise ValueError(f"Rating must be between 1 and 100. Got: {value}")
        return value

document_input = """
Internal Compliance Memo (June 2026):
Supplier SUP-9982 was audited. While the official checklist was dated 2026-08-15,
the operations team scored them a failing score of 0 due to critical environmental violations.
"""

# The client handles retry loops if validator checks fail
validated_log = client.chat.completions.create(
    model="gpt-4o",
    response_model=SupplierAuditLog,
    messages=[{"role": "user", "content": document_input}],
    max_retries=3
)
print(validated_log.model_dump_json(indent=2))
```

---

### 3. Supply Chain Event Extraction Schema

This JSON schema uses a polymorphic configuration (`oneOf`) to validate extracted parameters based on the identified event type:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "SupplyChainDisruptionEvent",
  "description": "Standardized schema for tracking supply chain disruption and risk events.",
  "type": "object",
  "required": ["event_id", "event_type", "confidence_score", "source_timestamp", "arguments"],
  "properties": {
    "event_id": {
      "type": "string",
      "pattern": "^EVT-[A-Z0-9]{8}$",
      "description": "Unique identifier format: EVT-XXXXXXXX."
    },
    "event_type": {
      "type": "string",
      "enum": ["ShipmentDelay", "SupplierInsolvency", "ForceMajeure", "FacilityHalt", "TariffChange"]
    },
    "confidence_score": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0
    },
    "source_timestamp": {
      "type": "string",
      "format": "date-time"
    },
    "arguments": { "type": "object" }
  },
  "oneOf": [
    {
      "properties": {
        "event_type": { "const": "ShipmentDelay" },
        "arguments": {
          "type": "object",
          "required": ["carrier", "consignment_id", "delay_duration_days", "reason"],
          "properties": {
            "carrier": { "type": "string" },
            "consignment_id": { "type": "string" },
            "origin": { "type": "string" },
            "destination": { "type": "string" },
            "delay_duration_days": { "type": "integer", "minimum": 1 },
            "reason": { "type": "string" }
          }
        }
      }
    },
    {
      "properties": {
        "event_type": { "const": "SupplierInsolvency" },
        "arguments": {
          "type": "object",
          "required": ["supplier_name", "legal_action", "filing_date"],
          "properties": {
            "supplier_name": { "type": "string" },
            "legal_action": {
              "type": "string",
              "enum": ["Bankruptcy", "Liquidation", "Restructuring", "Receivership"]
            },
            "filing_date": { "type": "string", "format": "date" },
            "jurisdiction": { "type": "string" },
            "affected_product_categories": {
              "type": "array",
              "items": { "type": "string" }
            }
          }
        }
      }
    },
    {
      "properties": {
        "event_type": { "const": "ForceMajeure" },
        "arguments": {
          "type": "object",
          "required": ["declaring_entity", "cause", "effective_date"],
          "properties": {
            "declaring_entity": { "type": "string" },
            "cause": { "type": "string" },
            "location_affected": { "type": "string" },
            "effective_date": { "type": "string", "format": "date" },
            "estimated_end_date": { "type": "string", "format": "date" }
          }
        }
      }
    },
    {
      "properties": {
        "event_type": { "const": "FacilityHalt" },
        "arguments": {
          "type": "object",
          "required": ["operator", "facility_location", "disruption_type", "start_date"],
          "properties": {
            "operator": { "type": "string" },
            "facility_location": { "type": "string" },
            "disruption_type": {
              "type": "string",
              "enum": ["Strike", "Accident", "Utility_Outage", "Maintenance", "Natural_Disaster"]
            },
            "start_date": { "type": "string", "format": "date" },
            "expected_restart_date": { "type": "string", "format": "date" }
          }
        }
      }
    },
    {
      "properties": {
        "event_type": { "const": "TariffChange" },
        "arguments": {
          "type": "object",
          "required": ["originating_country", "target_country", "tariff_percentage_increase"],
          "properties": {
            "originating_country": { "type": "string" },
            "target_country": { "type": "string" },
            "tariff_percentage_increase": { "type": "number" },
            "affected_goods": {
              "type": "array",
              "items": { "type": "string" }
            },
            "effective_date": { "type": "string", "format": "date" }
          }
        }
      }
    }
  ]
}
```

---

### 4. Step-by-Step Inter-Annotator Agreement Calculation

When building gold-standard validation sets, annotations are evaluated for consistency using Cohen's Kappa ($\kappa$):

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

Where:
*   $p_o$ is the observed proportional agreement between annotators.
*   $p_e$ is the expected probability of agreement by chance.

#### Calculation Example
Two annotators classify $100$ supply chain risk documents into three categories: **No Event (NE)**, **Facility Halt (FH)**, or **Shipment Delay (SD)**.

| Annotator A | Annotator B: NE | Annotator B: FH | Annotator B: SD | Marginal Row Totals |
| :--- | :---: | :---: | :---: | :---: |
| **NE** | 42 | 3 | 5 | **50** |
| **FH** | 2 | 20 | 3 | **25** |
| **SD** | 4 | 1 | 20 | **25** |
| **Marginal Col Totals** | **48** | **24** | **28** | **Total ($N$): 100** |

#### Step 1: Compute Observed Agreement ($p_o$)
Sum the diagonal values where both annotators agree:

$$p_o = \frac{42 + 20 + 20}{100} = 0.82$$

#### Step 2: Compute Expected Agreement by Chance ($p_e$)
Sum the products of the marginal row and column proportions for each class:

$$P(\text{Chance Agreement}) = \sum P(A_{\text{Class}}) \times P(B_{\text{Class}})$$

*   **For Class NE:** $\frac{50}{100} \times \frac{48}{100} = 0.50 \times 0.48 = 0.240$
*   **For Class FH:** $\frac{25}{100} \times \frac{24}{100} = 0.25 \times 0.24 = 0.060$
*   **For Class SD:** $\frac{25}{100} \times \frac{28}{100} = 0.25 \times 0.28 = 0.070$

$$p_e = 0.240 + 0.060 + 0.070 = 0.370$$

#### Step 3: Compute Cohen's Kappa ($\kappa$)

$$\kappa = \frac{0.820 - 0.370}{1 - 0.370} = \frac{0.450}{0.630} \approx 0.714$$

> **Evaluation:** A score of $0.714$ indicates moderate-to-strong agreement. Values above $0.80$ are generally preferred for reliable gold-standard benchmarks.

---

### 5. Benchmark Dataset Reference Inventory

A curated set of **53 foundational events** across supply chain disruptions, logistics failures, trade disputes, labor disputes, cyber incidents, and natural disasters, compiled as a benchmark repository.

#### Category 1: Logistics & Port Congestion (ShipmentDelay)

1. [2021 Suez Canal Obstruction](https://en.wikipedia.org/wiki/2021_Suez_Canal_obstruction)
2. [2024 Baltimore Bridge Collapse](https://en.wikipedia.org/wiki/Francis_Scott_Key_Bridge_collapse)
3. [2023–2024 Red Sea Crisis](https://en.wikipedia.org/wiki/Red_Sea_crisis)
4. [2021–2022 Global Supply Chain Crisis](https://en.wikipedia.org/wiki/2021%E2%80%932022_global_supply_chain_crisis)
5. [2022 Shanghai COVID-19 Outbreak (Port Lockdowns)](https://en.wikipedia.org/wiki/2022_Shanghai_COVID-19_outbreak)
6. [2020–2023 North American Drought (Mississippi Low Water)](https://en.wikipedia.org/wiki/2020%E2%80%932023_North_American_drought)
7. [2022 European Drought (Rhine River Low Water)](https://en.wikipedia.org/wiki/2022_European_drought)
8. [November 2021 Pacific Northwest Floods (Vancouver Rail Cutoff)](https://en.wikipedia.org/wiki/November_2021_Pacific_Northwest_floods)
9. [2021 United Kingdom Fuel Shortage](https://en.wikipedia.org/wiki/2021_United_Kingdom_fuel_shortage)
10. [Saint Lawrence Seaway (North American Shipping Corridor)](https://en.wikipedia.org/wiki/Saint_Lawrence_Seaway)
11. [Panama Canal (Global Shipping Chokepoint)](https://en.wikipedia.org/wiki/Panama_Canal)

#### Category 2: Labor Disputes & Work Stoppages

12. [2023 US United Auto Workers Strike](https://en.wikipedia.org/wiki/2023_United_Auto_Workers_strike)
13. [2024 Canada Rail Lockout](https://en.wikipedia.org/wiki/2024_Canada_railway_dispute)
14. [2023 German Transport Strikes](https://en.wikipedia.org/wiki/2023_German_public_transport_strike)
15. [2023 French Pension Reform Protests (Port Blockades)](https://en.wikipedia.org/wiki/2023_French_pension_reform_protests)
16. [2022–2024 United Kingdom Railway Strikes](https://en.wikipedia.org/wiki/2022%E2%80%932024_United_Kingdom_railway_strikes)
17. [Tesla Strike in Sweden (Nordic Labor Dispute)](https://en.wikipedia.org/wiki/Tesla_and_trade_unions)
18. [2024 United States Port Strike](https://en.wikipedia.org/wiki/2024_United_States_port_strike)
19. [2021–2022 Peruvian Mining Protests](https://en.wikipedia.org/wiki/2021%E2%80%932022_Peruvian_mining_protests)
20. [1997 United Parcel Service Strike](https://en.wikipedia.org/wiki/1997_United_Parcel_Service_strike)

#### Category 3: Natural Disasters & Environmental Events (ForceMajeure)

21. [2011 Tohoku Earthquake and Tsunami](https://en.wikipedia.org/wiki/2011_T%C5%8Dhoku_earthquake_and_tsunami)
22. [2021 Texas Winter Storm (Uri)](https://en.wikipedia.org/wiki/2021_Texas_power_crisis)
23. [2023 East Palestine Train Derailment (Chemical Release)](https://en.wikipedia.org/wiki/2023_Ohio_train_derailment)
24. [2022 European Heatwave](https://en.wikipedia.org/wiki/2022_European_heatwaves)
25. [2011 Thailand Floods](https://en.wikipedia.org/wiki/2011_Thailand_floods)
26. [2023 Canadian Wildfires](https://en.wikipedia.org/wiki/2023_Canadian_wildfires)
27. [2010 Eyjafjallajökull Volcanic Eruption](https://en.wikipedia.org/wiki/2010_eruptions_of_Eyjafjallaj%C3%B6kull)
28. [2024 Hurricane Beryl](https://en.wikipedia.org/wiki/Hurricane_Beryl)
29. [2022 Australian Floods](https://en.wikipedia.org/wiki/2022_eastern_Australia_floods)
30. [2024 Taiwan Earthquake](https://en.wikipedia.org/wiki/2024_Hualien_earthquake)
31. [2022 Cyclone Sitrang](https://en.wikipedia.org/wiki/Cyclone_Sitrang)

#### Category 4: Factory & Production Shutdowns (FacilityHalt)

32. [2020–2023 Global Semiconductor Shortage](https://en.wikipedia.org/wiki/2020%E2%80%932023_global_chip_shortage)
33. [2020 Beirut Explosion (Port Facility Destruction)](https://en.wikipedia.org/wiki/2020_Beirut_explosion)
34. [2024 Boeing 737 MAX Groundings](https://en.wikipedia.org/wiki/2024_Boeing_737_MAX_groundings)
35. [WannaCry Ransomware Attack (Global Assembly Line Halts)](https://en.wikipedia.org/wiki/WannaCry_ransomware_attack)
36. [Colonial Pipeline Cyberattack (Energy Infrastructure Shutdown)](https://en.wikipedia.org/wiki/Colonial_Pipeline_cyberattack)
37. [Deepwater Horizon Explosion](https://en.wikipedia.org/wiki/Deepwater_Horizon_explosion)
38. [Tesla Giga Berlin (Gigafactory Berlin-Brandenburg Outages)](https://en.wikipedia.org/wiki/Gigafactory_Berlin-Brandenburg)
39. [2019 Abqaiq–Khurais Drone Attack (Saudi Oil Facility Outage)](https://en.wikipedia.org/wiki/2019_Abqaiq%E2%80%93Khurais_attack)
40. [Rana Plaza Building Collapse (Industrial Supply Chain Halt)](https://en.wikipedia.org/wiki/Rana_Plaza_collapse)
41. [2024 CrowdStrike Cybersecurity Incident (Global Systems Outage)](https://en.wikipedia.org/wiki/2024_CrowdStrike_incident)
42. [1984 Bhopal Gas Leak Disaster](https://en.wikipedia.org/wiki/Bhopal_disaster)

#### Category 5: Regulatory, Geopolitical & Trade Policy (TariffChange)

43. [2018–2019 US–China Trade War](https://en.wikipedia.org/wiki/China%E2%80%93United_States_trade_war)
44. [Tariffs in the First Trump Administration](https://en.wikipedia.org/wiki/Tariffs_in_the_first_Trump_administration)
45. [2022 Russia–Ukraine Conflict Sanctions](https://en.wikipedia.org/wiki/International_sanctions_during_the_Russo-Ukrainian_War)
46. [CHIPS and Science Act (US Semiconductor Trade Policy)](https://en.wikipedia.org/wiki/CHIPS_and_Science_Act)
47. [Rare Earths Trade Dispute (China Export Restrictions)](https://en.wikipedia.org/wiki/Rare_earths_trade_dispute)
48. [2021 Brexit Border Implementations](https://en.wikipedia.org/wiki/Brexit_transition_period)
49. [2022 Uyghur Forced Labor Prevention Act (UFLPA)](https://en.wikipedia.org/wiki/Uyghur_Forced_Labor_Prevention_Act)
50. [EU Carbon Border Adjustment Mechanism (CBAM)](https://en.wikipedia.org/wiki/EU_Carbon_Border_Adjustment_Mechanism)
51. [EU Corporate Sustainability Due Diligence Directive (CSDDD)](https://en.wikipedia.org/wiki/Corporate_Sustainability_Due_Diligence_Directive)
52. [US Bureau of Industry and Security Entity List](https://en.wikipedia.org/wiki/Entity_List)
53. [2002 United States Steel Tariff](https://en.wikipedia.org/wiki/2002_United_States_steel_tariff)

---
