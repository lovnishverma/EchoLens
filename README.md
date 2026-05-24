---
title: EchoLensLite
emoji: 👁️
colorFrom: purple
colorTo: gray
sdk: docker
pinned: false
license: apache-2.0
---

# 👁️ EchoLens Lite — Edge-Optimized Image Captioning with Audio Synthesis

> An end-to-end, accessibility-first pipeline that accepts an image and autonomously generates both a natural language caption and a synthesized spoken audio description — powered by quantized TFLite models for low-latency cloud inference.

[![Hugging Face Space](https://img.shields.io/badge/🤗%20HuggingFace-EchoLensLite-blue)](https://huggingface.co/spaces/LovnishVerma/EchoLensLite)
[![Kaggle Pre-Training](https://img.shields.io/badge/Kaggle-Flickr30k%20Pretraining-20BEFF?logo=kaggle)](https://www.kaggle.com/code/princelv84/training-notebook-echolens-ultimate-image-caption)
[![Kaggle Fine-Tuning](https://img.shields.io/badge/Kaggle-COCO%202017%20Finetuning-20BEFF?logo=kaggle)](https://www.kaggle.com/code/princelv84/coco-2017-dataset)
[![Built with TFLite](https://img.shields.io/badge/TFLite-Quantized-orange?logo=tensorflow)](https://www.tensorflow.org/lite)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange?logo=tensorflow)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-Educational%20%2F%20Research-green)](#license)

---

## Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [System Capabilities](#system-capabilities)
- [Architecture & Model Design](#architecture--model-design)
- [Training Protocol](#training-protocol)
  - [Stage 1 — Pre-Training on Flickr30k](#stage-1--pre-training-on-flickr30k)
  - [Stage 2 — Fine-Tuning on MS-COCO 2017](#stage-2--fine-tuning-on-ms-coco-2017)
- [Model Artifacts](#model-artifacts)
- [Edge Inference Pipeline](#edge-inference-pipeline)
- [Data Pipeline Details](#data-pipeline-details)
- [Repository Structure](#repository-structure)
- [Local Execution](#local-execution)
- [Deployment Architecture](#deployment-architecture)
- [Known Engineering Solutions](#known-engineering-solutions)
- [Dependencies](#dependencies)
- [Datasets & Credits](#datasets--credits)
- [Author](#author)
- [License](#license)

---

## Overview

**EchoLens Lite** is an accessibility-first image captioning system designed for visually impaired users. It takes any uploaded image and produces:

1. A concise, grammatically correct natural language description of the scene.
2. A synthesized `.mp3` audio file that reads the caption aloud via Google Text-to-Speech (gTTS).

The entire inference stack fits within **~22 MB** of model weight storage — achieved through 8-bit dynamic range quantization of a full EfficientNetB0 + Transformer Encoder-Decoder architecture. This makes EchoLens Lite viable for constrained cloud environments, Hugging Face Spaces free tiers, and potential edge hardware deployment.

---

## Live Demo

Try the hosted inference interface directly in your browser:

**[🚀 Launch EchoLens Lite on Hugging Face Spaces](https://huggingface.co/spaces/LovnishVerma/EchoLensLite)**

Upload any image, click **Generate Description**, and receive both a text caption and a playable audio description within seconds.

---

## System Capabilities

| Capability | Detail |
|---|---|
| Input | Any RGB image (JPEG, PNG, etc.) uploaded via browser or camera |
| Caption Output | Natural language sequence up to 25 tokens |
| Audio Output | `.mp3` via gTTS, auto-playable in the Gradio UI |
| Model Footprint | ~22 MB total (quantized TFLite) |
| Inference Latency | Low-latency, suitable for cloud-hosted free tiers |
| Accessibility Target | Visually impaired users (WCAG-aligned audio output) |
| Vocabulary Size | 10,000 tokens |
| Image Input Size | 299×299×3 (normalized to `[0, 1]`) |

---

## Architecture & Model Design

EchoLens Lite implements a decoupled **CNN + Transformer Encoder-Decoder** topology, split into two independently quantized TFLite models for inference.

```
Image (299×299×3)
       │
       ▼
[ EfficientNetB0 CNN Backbone ]
  └─ ImageNet pre-trained weights
  └─ Top layers removed (feature extractor only)
  └─ Output reshaped: (batch, num_patches, 1280)
       │
       ▼
[ Transformer Encoder Block ]
  └─ 1-Head Self-Attention
  └─ Embedding dim: 512
  └─ Feed-forward dim: 512
  └─ Output: Image Feature Tensor (batch, num_patches, 512)
       │
       ├─────────────────────────────────────────────────────────┐
       │                                                         │
       ▼                                                         │
Sequence (<start> token)                                         │
       │                                                         │
       ▼                                                         │
[ Transformer Decoder Block ] ◄───── Cross-Attention ───────────┘
  └─ 2-Head Cross-Attention (image features + token sequence)
  └─ Embedding dim: 512
  └─ Feed-forward dim: 512
  └─ Output: Token Probability Distribution (batch, seq_len, vocab_size)
       │
       ▼
  Argmax → Next Token → Append → Loop until <end> or max length
```

### Component Specifications

**CNN Feature Extractor — EfficientNetB0**

- Weights initialized from ImageNet pre-training.
- `include_top=False`: classification head removed; spatial feature maps are retained.
- Output is reshaped from `(H, W, 1280)` to `(num_patches, 1280)` via a `Reshape` layer, yielding a sequence of visual "patch tokens" for the Transformer.
- During pre-training: fully frozen (only Transformer parameters update).
- During fine-tuning: the final 20 layers are unfrozen (`UNFREEZE_LAST_N = 20`) with a low learning rate (`1e-5`) to adapt domain-specific spatial features to the COCO captioning distribution without catastrophic forgetting.

**Transformer Encoder — Self-Attention over Visual Patches**

- 1 attention head operating over the spatial patch sequence produced by EfficientNetB0.
- Projects patch features into a 512-dimensional embedding space.
- Dense feed-forward sublayer with dimension 512 and residual connection.
- Produces the encoded image representation passed to the decoder at every decoding step.

**Transformer Decoder — Autoregressive Token Generation**

- 2 attention heads for cross-attention between the current token sequence and the encoded image representation.
- Embedding dimension: 512; feed-forward dimension: 512.
- Uses a causal mask on the sequence (padding tokens masked via `tf.math.not_equal(x, 0)`).
- At inference, probabilities for reserved tokens (`pad`, `unk`, `<start>`, `<end>`) are explicitly zeroed before `argmax` to prevent degenerate outputs.
- Generation terminates on `<end>` token or at `SEQ_LENGTH - 1 = 24` tokens.

**Text Vectorization**

- Implemented with Keras `TextVectorization` layer.
- Maximum vocabulary: 10,000 tokens (`VOCAB_SIZE = 10000`).
- Output sequence length: 25 (`SEQ_LENGTH = 25`).
- Custom standardization: lowercases input and strips punctuation (excluding `<` and `>` to preserve special tokens).
- Vocabulary serialized to `vectorization_layer_state.pkl` (Protocol 4) for cross-environment compatibility.

---

## Training Protocol

The model uses a **two-stage transfer learning paradigm** to maximize generalization while preserving previously learned visual semantics.

### Stage 1 — Pre-Training on Flickr30k

**Dataset:** [Flickr30k](https://www.kaggle.com/datasets/princelv84/flickr30k)
**Notebook:** [Kaggle — EchoLens Pre-Training](https://www.kaggle.com/code/princelv84/training-notebook-echolens-ultimate-image-caption)

- ~31,000 images, 5 human-annotated captions each.
- Captions filtered to 5–25 tokens; shorter or longer captions are dropped.
- 80/20 train/validation split with shuffling.
- `<start>` and `<end>` tokens prepended/appended to every caption during loading.

**Training Configuration:**

| Hyperparameter | Value |
|---|---|
| Image size | 299×299 |
| Vocabulary size | 10,000 |
| Sequence length | 25 |
| Embedding dim | 512 |
| Feed-forward dim | 512 |
| Batch size | 32 |
| Epochs | 15 (with early stopping, patience=3) |
| Loss | Sparse Categorical Crossentropy |
| Optimizer | Adam + Linear Warmup LR Schedule |
| Post-warmup LR | 1e-4 |
| Warmup steps | `num_train_steps // 15` |

**Data Augmentation (Training Only):**

- `RandomFlip("horizontal")`
- `RandomRotation(0.2)` — 20% rotation range
- `RandomContrast(0.3)` — 30% contrast jitter

**TF Data Pipeline:** Images cached in RAM after first pass (`.cache()`) for accelerated subsequent epochs.

**Callbacks:**
- `ModelCheckpoint` — saves best model by `val_loss`
- `EarlyStopping` — patience of 3 epochs, restores best weights

**Output Artifacts (Stage 1):**
- `caption_model.keras` — full Keras model (>200 MB, includes optimizer state)
- `vectorization_layer_state.pkl` — vocabulary mapping

---

### Stage 2 — Fine-Tuning on MS-COCO 2017

**Dataset:** [MS-COCO 2017](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)
**Notebook:** [Kaggle — EchoLens COCO Fine-Tuning](https://www.kaggle.com/code/princelv84/coco-2017-dataset)

MS-COCO is larger and more diverse than Flickr30k, covering 80 object categories across complex real-world scenes. Fine-tuning on COCO substantially improves caption quality and vocabulary coverage for everyday imagery.

- Images used Full 118k ( if you want you can limit Up to 60,000 training images and 5,000 validation images capped via `MAX_COCO_TRAIN / MAX_COCO_VAL`).
- Annotations loaded from COCO JSON format (`captions_train2017.json`, `captions_val2017.json`).
- Each image padded or truncated to exactly 5 captions (`NUM_CAPTIONS_PER_IMAGE = 5`) to produce rectangular tensors for efficient batching.
- The pre-trained Flickr30k vocabulary is reused — no vocabulary re-adaptation.

**Fine-Tuning Configuration:**

| Hyperparameter | Value |
|---|---|
| Base LR | 1e-5 (lower than Stage 1 to protect pre-trained weights) |
| Epochs | 10 (with early stopping, patience=3) |
| CNN unfrozen layers | Last 20 of EfficientNetB0 |
| LR Reduction | `ReduceLROnPlateau` factor=0.5, patience=2, min=1e-7 |

**Data Augmentation (Fine-Tuning):**

Slightly more conservative than Stage 1 to avoid destroying ImageNet features:
- `RandomFlip("horizontal")`
- `RandomRotation(0.1)` — 10% rotation range
- `RandomContrast(0.2)` — 20% contrast jitter
- `RandomBrightness(factor=0.1)` — 10% brightness jitter

**Weight Transfer Strategy:**

Due to a Keras 2 / Keras 3 serialization incompatibility between training environments, the architecture is **programmatically reconstructed from scratch** in the fine-tuning notebook and pre-trained weights are injected purely via functional layer name-mapping (`load_weights(..., by_name=True, skip_mismatch=True)`). This avoids `Unknown object: 'Functional'` errors when loading a `.keras` file across different Keras versions.

**NumPy Compatibility Fix:**

The vocabulary `.pkl` file was saved under NumPy 2.x, which uses the `numpy._core` module path. Kaggle's environment (NumPy 1.23.5) uses `numpy.core`. A custom `NumpyCompatUnpickler` shim transparently rewrites the module path at load time — no reinstalls required. The vocabulary is then re-saved with `pickle.dump(..., protocol=4)` for future portability.

**Callbacks (Fine-Tuning):**
- `ModelCheckpoint` — saves best weights (`save_weights_only=True` for subclassed model compatibility)
- `EarlyStopping` — patience of 3 epochs
- `ReduceLROnPlateau` — halves LR when val_loss plateaus for 2 epochs

---

## Model Artifacts

After fine-tuning, the full Keras model (>200 MB) is decomposed into two functional sub-models and compressed via **8-bit dynamic range quantization** using the TFLite converter. This replaces 32-bit float weights with 8-bit integers, achieving ~4x size reduction with negligible accuracy loss.

Both TFLite models are compiled with:
- `tf.lite.Optimize.DEFAULT` (dynamic range quantization)
- `tf.lite.OpsSet.SELECT_TF_OPS` (required for custom TF ops used in the Transformer)

| Artifact | Description | Quantization | Size |
|---|---|---|---|
| `echolens_encoder_quantized.tflite` | EfficientNetB0 CNN + Transformer Encoder | 8-bit dynamic | 7.01 MB |
| `echolens_decoder_quantized.tflite` | Transformer Decoder (autoregressive) | 8-bit dynamic | 14.4 MB |
| `vectorization_layer_state.pkl` | 10k token vocabulary mapping | Protocol-4 | 647 KB |

**Total inference footprint: ~22 MB** (vs >200 MB for the full `.keras` format).

---

## Edge Inference Pipeline

The inference pipeline in `app.py` is designed to **exactly replicate** the training preprocessing to prevent train-inference mismatch.

**Step 1 — Image Preprocessing**

```python
img = tf.convert_to_tensor(image_numpy, dtype=tf.uint8)      # uint8 [0, 255]
img = tf.image.resize(img, (299, 299))                         # float32 [0, 255]
img = tf.image.convert_image_dtype(img, tf.float32)            # float32 [0, 1]
img = tf.expand_dims(img, axis=0)                              # shape: (1, 299, 299, 3)
```

The critical detail is using `tf.image.convert_image_dtype` (which maps `[0, 255] → [0, 1]`) rather than manual division, matching the training `decode_and_resize()` function exactly.

**Step 2 — TFLite Encoding**

The encoder interpreter accepts `(1, 299, 299, 3)` float32 input and produces the encoded image feature tensor. Tensor shapes are dynamically resized before each invocation to support flexible batch/sequence dimensions.

**Step 3 — Autoregressive Decoding**

```
decoded_caption = "<start> "
for step in range(SEQ_LENGTH - 1):
    tokenize current decoded_caption → (1, SEQ_LENGTH-1) int32
    run decoder TFLite → (1, SEQ_LENGTH-1, VOCAB_SIZE) predictions
    find position of last non-zero token
    mask special token indices [0,1,2,3] to 0
    argmax → next token index → look up in vocabulary
    if token == "<end>" or "": break
    append token to decoded_caption
```

**Step 4 — Text-to-Speech**

The final caption string is passed to `gTTS(text=caption, lang="en", slow=False)`, which synthesizes an `.mp3` file stored in a temporary path and returned to Gradio for auto-playback.

**Decoder Tensor Resolution**

To handle TFLite model tensor indices robustly (which can vary across conversion runs), input tensors are resolved by name hint (`"sequence"` → int32, `"encoded"` → float32) with a dtype fallback, ensuring the correct tensor is always mapped regardless of index ordering.

---

## Data Pipeline Details

### Flickr30k Caption Loading

- CSV/TXT files are auto-detected via dynamic path finding (`os.walk`).
- Lines with fewer than 5 or more than 25 tokens are filtered out.
- Pipe-delimited files (`|`) are normalized to commas before parsing.
- Each caption is wrapped: `"<start> " + caption + " <end>"`.
- Images with any invalid captions are fully excluded from training.

### COCO 2017 Annotation Loading

- Annotations parsed from JSON format (`image_id → file_name` mapping built from `data["images"]`).
- Only annotations whose image files exist on disk are included.
- Token length filtering: 5–25 tokens.
- Caption lists per image are padded (by repeating the last caption) or truncated to exactly 5, ensuring rectangular tensor batches.

### Vectorization

```python
strip_chars = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""" # < and > preserved
vectorization = TextVectorization(
    max_tokens=10000,
    output_mode="int",
    output_sequence_length=25,
    standardize=custom_standardization,  # lowercase + strip punctuation
)
```

The same `custom_standardization` function and `strip_chars` pattern is used identically in training, fine-tuning, and inference — any divergence here would corrupt the vocabulary mapping.

---

## Repository Structure

```
EchoLensLite/
├── app.py                               # Gradio web UI + full inference pipeline
├── config.yaml                          # Hyperparameter definitions
├── Dockerfile                           # Hugging Face Spaces containerization
├── requirements.txt                     # Python runtime dependencies
├── vectorization_layer_state.pkl        # Serialized 10k token vocabulary (Protocol 4)
├── echolens_encoder_quantized.tflite    # Quantized CNN + Transformer Encoder (~7 MB)
└── echolens_decoder_quantized.tflite    # Quantized Transformer Decoder (~14 MB)
```

**Training Notebooks (Kaggle):**

```
kaggle/
├── training_notebook_echolens_ultimate_image_caption.py   # Stage 1: Flickr30k pre-training
└── coco_2017_dataset.py                                    # Stage 2: COCO fine-tuning + TFLite export
```

---

## Local Execution

### Prerequisites

- Python 3.10
- pip

### Setup

```bash
# 1. Clone the repository
git clone https://huggingface.co/spaces/LovnishVerma/EchoLensLite
cd EchoLensLite

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Gradio interface
python app.py
```

Navigate to `http://localhost:7860` in your browser. The interface allows you to upload an image via file picker or camera capture.

### Configuration

All key hyperparameters are centralized in `config.yaml`:

```yaml
image_preprocessing:
  IMAGE_SIZE: [299, 299]

training:
  VOCAB_SIZE: 10000
  SEQ_LENGTH: 25

paths:
  vocab_file: "vectorization_layer_state.pkl"
```

To use a different vocabulary file or adjust sequence length, modify `config.yaml` — no code changes required.

---

## Deployment Architecture

EchoLens Lite is containerized for seamless deployment on Hugging Face Docker Spaces.

**Dockerfile Summary:**

```dockerfile
FROM python:3.10-slim

# System deps: OpenCV (libgl1), glib (libglib2.0-0), audio rendering (ffmpeg)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 ffmpeg

# Non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
USER user

WORKDIR /app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .

EXPOSE 7860
CMD ["python", "app.py"]
```

Key deployment decisions:

- **Base image:** `python:3.10-slim` — minimal footprint, no unnecessary system packages.
- **libgl1** replaces the deprecated `libgl1-mesa-glx` for current Debian slim images.
- **Non-root execution:** UID 1000 mapped to `/home/user` satisfies Hugging Face container security requirements.
- **Port 7860:** Default Gradio port, auto-exposed and routed by the HF Spaces proxy.
- **No GPU required:** All inference runs on CPU via TFLite; the quantized models are fast enough for real-time use without a GPU.

---

## Known Engineering Solutions

Several non-obvious issues were encountered and solved during development. They are documented here to assist future contributors.

### Keras 2 / Keras 3 Serialization Incompatibility

**Problem:** The pre-training notebook (Keras 3 environment) saved `caption_model.keras`. The fine-tuning notebook (Kaggle's TF-bundled Keras 2) could not load it, raising `ValueError: Unknown object: 'Functional'`.

**Solution:** In the fine-tuning notebook, the model architecture is fully reconstructed from scratch and Flickr30k weights are transferred purely via `load_weights(..., by_name=True, skip_mismatch=True)` — bypassing the broken serialization entirely.

### NumPy 1.x / 2.x Pickle Incompatibility

**Problem:** The vocabulary `.pkl` was saved under NumPy 2.x (which uses `numpy._core` internally). Kaggle's environment uses NumPy 1.23.5 (`numpy.core`), causing `ModuleNotFoundError` during unpickling.

**Solution:** A `NumpyCompatUnpickler` subclass rewrites `numpy._core` → `numpy.core` in `find_class()` at load time. The vocabulary is then immediately re-saved with `protocol=4` for forward compatibility.

### TFLite Dynamic Tensor Indexing

**Problem:** TFLite model input tensor indices can shift between conversion runs, making hardcoded index lookups fragile.

**Solution:** Tensor indices are resolved at startup by matching name hints (`"sequence"`, `"encoded"`) against `get_input_details()`, with a dtype fallback (`np.int32` / `np.float32`). This is robust to reordering.

### COCO Variable Caption Count

**Problem:** MS-COCO images have variable numbers of captions (typically 3–7), which breaks fixed-shape tensor batching.

**Solution:** All caption lists are padded (by repeating the last caption) or truncated to exactly `NUM_CAPTIONS_PER_IMAGE = 5`, producing rectangular tensors throughout the pipeline.

### Decoder Position Tracking at Inference

**Problem:** The TFLite decoder receives the full padded sequence at each step, but only the position of the last non-zero (non-padding) token should be read for the next token prediction.

**Solution:** `num_tokens = np.count_nonzero(tokenized_np[0])` counts filled positions; `pos = max(0, num_tokens - 1)` reads from the correct position in the predictions tensor, replicating the training-time indexing behavior.

---

## Dependencies

```
tensorflow==2.19.0
gradio
gTTS
numpy
PyYAML
opencv-python-headless
```

The `opencv-python-headless` variant is used (rather than `opencv-python`) to avoid pulling in GUI display backends that are unnecessary in a server environment and can cause import failures in headless Docker containers.

---

## Datasets & Credits

| Dataset | Source | Usage |
|---|---|---|
| Flickr30k | [Kaggle — princelv84/flickr30k](https://www.kaggle.com/datasets/princelv84/flickr30k) | Stage 1 pre-training |
| MS-COCO 2017 | [Kaggle — awsaf49/coco-2017-dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) | Stage 2 fine-tuning |

**Dataset Citation:**

Young, P., Lai, A., Hodosh, M., & Hockenmaier, J. (2014). *From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions.* Transactions of the Association for Computational Linguistics, 2, 67–78.

Lin, T.-Y., et al. (2014). *Microsoft COCO: Common objects in context.* In European Conference on Computer Vision (ECCV).

---

## Author

**Lovnish Verma**

[🌐 Portfolio](https://lovnishverma.in) · [💻 GitHub](https://github.com/lovnishverma) · [🔗 LinkedIn](https://linkedin.com/in/lovnishverma) · [📊 Kaggle](https://www.kaggle.com/princelv84) · [🤗 Hugging Face](https://huggingface.co/LovnishVerma)

---

## License

This project is released for **educational and research use only**.

The model weights, training notebooks, and inference code are provided as-is. Dataset licenses (Flickr30k, MS-COCO) apply to their respective data. Please consult the original dataset authors for any commercial use queries.
