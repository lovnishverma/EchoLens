---
title: EchoLensLite
emoji: 👁️
colorFrom: purple
colorTo: gray
sdk: docker
pinned: false
---

# 👁️ EchoLens Lite — Edge-Optimized Image Captioning

fine tuning notebook url: https://www.kaggle.com/code/princelv84/coco-2017-dataset

> Upload any image → get an automatic caption + spoken audio description, powered by a quantized TFLite model running entirely in the cloud.

[![Hugging Face Space](https://img.shields.io/badge/🤗%20HuggingFace-EchoLensLite-blue)](https://huggingface.co/spaces/LovnishVerma/EchoLensLite)
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Training%20Notebook-20BEFF?logo=kaggle)](https://www.kaggle.com/code/princelv84/training-notebook-echolens-ultimate-image-caption)
[![Dataset](https://img.shields.io/badge/Kaggle-Flickr30k%20Dataset-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/princelv84/flickr30k)
[![Built with TFLite](https://img.shields.io/badge/TFLite-Quantized-orange?logo=tensorflow)](https://www.tensorflow.org/lite)

---

## 🚀 What It Does

EchoLens Lite takes any uploaded photo and produces:

- **A natural language caption** describing the image content
- **An audio description** (text-to-speech via gTTS) that reads the caption aloud — designed with accessibility in mind

It runs on quantized TFLite models (~22 MB total), making it fast and lightweight enough for edge and cloud deployment.

---

## 🧠 Model Architecture

The system is a two-stage pipeline:


```

Image → [EfficientNetB0 CNN + Transformer Encoder] → Image Features
↓
 token → [Transformer Decoder] → Caption tokens (autoregressive)

```

| Component | Details |
|---|---|
| **CNN Backbone** | EfficientNetB0 (ImageNet weights, frozen) |
| **Transformer Encoder** | 1-head self-attention, embed_dim=512 |
| **Transformer Decoder** | 2-head cross-attention, embed_dim=512, ff_dim=512 |
| **Vocabulary** | 10,000 tokens, trained on Flickr30k captions |
| **Input Size** | 299 × 299 × 3 |
| **Max Caption Length** | 25 tokens |
| **Quantization** | 8-bit dynamic range (TFLite) |

### Model Sizes

| File | Size |
|---|---|
| `echolens_encoder_quantized.tflite` | 6.31 MB |
| `echolens_decoder_quantized.tflite` | 15.3 MB |
| **Total** | **~22 MB** (vs. 200+ MB original `.keras`) |

---

## 📦 Training

The model was trained on the **Flickr30k dataset** (~31,000 images, 5 captions each).

| Hyperparameter | Value |
|---|---|
| Epochs | 15 (early stopping, patience=3) |
| Batch Size | 32 |
| Optimizer | Adam + warmup LR schedule (peak 1e-4) |
| Loss | Sparse Categorical Crossentropy |
| Data Split | 80% train / 20% validation |

**Full training notebook:** [Kaggle](https://www.kaggle.com/code/princelv84/training-notebook-echolens-ultimate-image-caption)

**Dataset:** [Flickr30k on Kaggle](https://www.kaggle.com/datasets/princelv84/flickr30k)

---

## 🗂️ Repository Structure


```

EchoLensLite/
├── app.py                              # Gradio inference app
├── config.yaml                         # Model hyperparameters
├── Dockerfile                          # HF Docker Space config
├── requirements.txt                    # Python dependencies
├── vectorization_layer_state.pkl       # Saved vocabulary (10k tokens)
├── echolens_encoder_quantized.tflite   # Quantized encoder (6.31 MB)
├── echolens_decoder_quantized.tflite   # Quantized decoder (15.3 MB)
└── training-notebook-echolens-ultimate-image-caption (kaggle).ipynb

```

---

## ⚙️ How Inference Works

1. **Preprocess** — image resized to 299×299, normalized to `[0, 1]` (matching training pipeline exactly via `tf.image.convert_image_dtype`)
2. **Encode** — TFLite encoder runs EfficientNetB0 + Transformer Encoder → image feature tensor
3. **Decode** — autoregressive loop: at each step, the partial caption is tokenized and fed with image features into the TFLite decoder; the highest-probability next token is selected
4. **Speak** — final caption is passed to gTTS and returned as an `.mp3` audio file

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Model Training | TensorFlow / Keras |
| Inference Runtime | TFLite (quantized) |
| Web UI | Gradio |
| Text-to-Speech | gTTS |
| Containerization | Docker |
| Deployment | Hugging Face Docker Spaces |

---

## 🔧 Run Locally

```bash
git clone [https://huggingface.co/spaces/LovnishVerma/EchoLensLite](https://huggingface.co/spaces/LovnishVerma/EchoLensLite)
cd EchoLensLite
pip install -r requirements.txt
python app.py

```

Then open `http://localhost:7860` in your browser.

---

## 👤 Author

**Lovnish Verma**

[Portfolio](https://lovnishverma.in) · [GitHub](https://github.com/lovnishverma) · [LinkedIn](https://linkedin.com/in/lovnishverma) · [Kaggle](https://www.kaggle.com/princelv84) · [Hugging Face](https://huggingface.co/LovnishVerma)

---

## 📄 License

This project is released for educational and research use.
Dataset credit: Flickr30k — Young et al., *From image descriptions to visual denotations*, TACL 2014.

