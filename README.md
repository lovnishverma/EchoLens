---
title: EchoLensLite
emoji: 👁️
colorFrom: purple
colorTo: gray
sdk: docker
pinned: false
---

# 👁️ EchoLens Lite — Edge-Optimized Image Captioning

> An end-to-end edge-optimized pipeline that accepts an image and autonomously generates both a natural language caption and a synthesized spoken audio description, utilizing quantized TFLite models for low-latency cloud inference.

[![Hugging Face Space](https://img.shields.io/badge/🤗%20HuggingFace-EchoLensLite-blue)](https://huggingface.co/spaces/LovnishVerma/EchoLensLite)
[![Kaggle Pre-Training](https://img.shields.io/badge/Kaggle-Flickr30k%20Pretraining-20BEFF?logo=kaggle)](https://www.kaggle.com/code/princelv84/training-notebook-echolens-ultimate-image-caption)
[![Kaggle Fine-Tuning](https://img.shields.io/badge/Kaggle-COCO%202017%20Finetuning-20BEFF?logo=kaggle)](https://www.kaggle.com/code/princelv84/coco-2017-dataset)
[![Built with TFLite](https://img.shields.io/badge/TFLite-Quantized-orange?logo=tensorflow)](https://www.tensorflow.org/lite)

---

## 🚀 System Capabilities

EchoLens Lite is designed for accessibility and edge-deployment efficiency. It processes uploaded media to produce:
1.  **Autoregressive Captions:** A natural language sequence describing the visual topography.
2.  **Audio Synthesis:** Text-to-speech conversion (via gTTS) outputting an `.mp3` for visually impaired accessibility.

The entire model footprint is compressed to ~22 MB via 8-bit dynamic range quantization, bypassing the >200MB overhead of standard Keras `.keras` or HDF5 formats.

---

## 🧠 Architecture & Model Design

The system implements a decoupled encoder-decoder Transformer topology.

```text
Image (299x299x3) 
       │
       ▼
[ EfficientNetB0 CNN Backbone (Frozen) ] 
       │
       ▼
[ Transformer Encoder (1-Head Self-Attention) ] ───> Image Feature Tensor
                                                            │
                                                            ▼
Sequence (<start>) ───> [ Transformer Decoder (2-Head Cross-Attention) ] ───> Token Probabilities

```

### Component Specifications

* **CNN Extractor:** `EfficientNetB0` utilizing ImageNet weights. The final 20 layers are unfrozen during the fine-tuning phase to adapt spatial feature maps to the captioning domain.
* **Transformer Encoder:** 1-head self-attention mechanism with an embedding space of 512 and a dense feed-forward dimension of 512.
* **Transformer Decoder:** Autoregressive generator with 2-head cross-attention mapping the encoded image representation and previous tokens to output probabilities.
* **Vocabulary:** Constrained to 10,000 distinct tokens, outputting sequences up to 25 tokens in length (`SEQ_LENGTH=25`).

### Artifact Footprint

| File | Quantization | Size |
| --- | --- | --- |
| `echolens_encoder_quantized.tflite` | 8-bit dynamic | 6.31 MB |
| `echolens_decoder_quantized.tflite` | 8-bit dynamic | 15.3 MB |
| `vectorization_layer_state.pkl` | Protocol-4 | 662 KB |

---

## 📦 Training Protocol & Dataset Processing

The model utilizes a two-stage training paradigm to maximize semantic density while avoiding catastrophic forgetting.

1. **Pre-Training:** Initialized on the **Flickr30k dataset** (~31,000 images, 5 captions each).
2. **Fine-Tuning:** Transferred and fine-tuned on the **MS-COCO 2017** dataset.
* **Pipeline Rectification:** To ensure rectangular tensors for efficient batching, MS-COCO annotations are dynamically padded or truncated to exactly 5 captions per image (`NUM_CAPTIONS_PER_IMAGE = 5`).
* **Augmentation:** Real-time data pipeline applies random horizontal flips, rotation (10%), contrast (20%), and brightness (10%) adjustments.
* **Optimization:** Sparse Categorical Crossentropy loss optimized via Adam with a custom linear-warmup Learning Rate Schedule.



*Note: During fine-tuning, to bypass Keras 2/Keras 3 serialization incompatibilities, the architecture is programmatically reconstructed and pre-trained weights are injected purely via functional layer name-mapping (`by_name=True`).*

---

## ⚙️ Edge Inference Execution

Inference strictly mimics the training pipeline to prevent degradation.

1. **Deterministic Preprocessing:** Uploaded numpy arrays are resized to `299x299` and converted to `tf.float32` arrays bounded strictly between `[0, 1]` utilizing `tf.image.convert_image_dtype`.
2. **Feature Encoding:** The quantized TFLite Encoder processes the normalized tensor into a fixed-dimensional spatial representation.
3. **Autoregressive Decoding:** The decoder iteratively predicts the next token. Probabilities for special tokens (pad, unk, start, end) are explicitly masked prior to `np.argmax` selection.

---

## 🗂️ Repository Structure

```text
EchoLensLite/
├── app.py                              # Gradio web UI and inference logic
├── config.yaml                         # Hyperparameter definitions (IMAGE_SIZE, VOCAB_SIZE)
├── Dockerfile                          # Deployment specifications for Hugging Face Spaces
├── requirements.txt                    # Runtime dependencies (tensorflow, gradio, gTTS, etc.)
├── vectorization_layer_state.pkl       # 10k token vocabulary mapping
├── echolens_encoder_quantized.tflite   # Encapsulated CNN + Transformer Encoder
└── echolens_decoder_quantized.tflite   # Encapsulated Transformer Decoder

```

---

## 🚢 Deployment Architecture

The application is containerized for seamless cloud deployment, explicitly configured for platforms like Hugging Face Spaces.

* **Base Image:** `python:3.10-slim`.
* **System Dependencies:** Integrates `libgl1`, `libglib2.0-0`, and `ffmpeg` to support OpenCV and audio rendering processes.
* **Execution Privilege:** Binds to a non-root user (UID 1000) mapped to `/home/user` for container security compliance.
* **Exposure:** Web UI served via Gradio on port `7860`.

---

## 🔧 Local Execution

To run the inference engine locally:

```bash
# 1. Clone the repository
git clone [https://huggingface.co/spaces/LovnishVerma/EchoLensLite](https://huggingface.co/spaces/LovnishVerma/EchoLensLite)
cd EchoLensLite

# 2. Install required dependencies
pip install -r requirements.txt

# 3. Initialize the Gradio interface
python app.py

```

Navigate to `http://localhost:7860` in your standard web browser.

---

## 👤 Author

**Lovnish Verma**

[Portfolio](https://lovnishverma.in) · [GitHub](https://github.com/lovnishverma) · [LinkedIn](https://linkedin.com/in/lovnishverma) · [Kaggle](https://www.kaggle.com/princelv84) · [Hugging Face](https://huggingface.co/LovnishVerma)

---

## 📄 License

This project is released for educational and research use.
Dataset credit: Flickr30k — Young et al., *From image descriptions to visual denotations*, TACL 2014.

