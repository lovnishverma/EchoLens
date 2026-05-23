"""
EchoLens: Edge-Optimized Image Captioning Pipeline
--------------------------------------------------
Inference using quantized TFLite models.
Preprocessing EXACTLY matches the training pipeline.
"""

import os
import re
import tempfile
import numpy as np
import tensorflow as tf
from keras.layers import TextVectorization
import pickle
import yaml
import gradio as gr
from gtts import gTTS

os.environ["KERAS_BACKEND"] = "tensorflow"

# ==========================================
# 1. CONFIGURATION  (must match training)
# ==========================================
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

IMAGE_SIZE  = tuple(config["image_preprocessing"]["IMAGE_SIZE"])   # (299, 299)
SEQ_LENGTH  = config["training"]["SEQ_LENGTH"]                      # 25
VOCAB_SIZE  = config["training"]["VOCAB_SIZE"]                      # 10000
VOCAB_PATH  = config["paths"]["vocab_file"]

print(f"Config loaded → IMAGE_SIZE={IMAGE_SIZE}, SEQ_LENGTH={SEQ_LENGTH}, VOCAB_SIZE={VOCAB_SIZE}")

# ==========================================
# 2. VECTORIZATION  (must match training)
# ==========================================
# Exact same strip_chars / standardization as training
strip_chars = r"!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~".replace("<", "").replace(">", "")

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

vectorization = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQ_LENGTH,
    standardize=custom_standardization,
)

with open(VOCAB_PATH, "rb") as f:
    vectorization_loaded_data = pickle.load(f)

vectorization.set_vocabulary(vectorization_loaded_data["vocab"])

vocab       = vectorization.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))
max_decoded_sentence_length = SEQ_LENGTH - 1

print(f"Vocabulary loaded → {len(vocab)} tokens")

# ==========================================
# 3. LOAD TFLITE MODELS
# ==========================================
print("Loading TFLite models...")

encoder_interpreter = tf.lite.Interpreter(model_path="echolens_encoder_quantized.tflite")
encoder_interpreter.allocate_tensors()

decoder_interpreter = tf.lite.Interpreter(model_path="echolens_decoder_quantized.tflite")
decoder_interpreter.allocate_tensors()

# ---- Print tensor details once at startup for debugging ----
print("\n[Encoder] Input tensors:")
for t in encoder_interpreter.get_input_details():
    print(f"  index={t['index']}  name={t['name']}  shape={t['shape']}  dtype={t['dtype']}")
print("[Encoder] Output tensors:")
for t in encoder_interpreter.get_output_details():
    print(f"  index={t['index']}  name={t['name']}  shape={t['shape']}  dtype={t['dtype']}")

print("\n[Decoder] Input tensors:")
for t in decoder_interpreter.get_input_details():
    print(f"  index={t['index']}  name={t['name']}  shape={t['shape']}  dtype={t['dtype']}")
print("[Decoder] Output tensors:")
for t in decoder_interpreter.get_output_details():
    print(f"  index={t['index']}  name={t['name']}  shape={t['shape']}  dtype={t['dtype']}")

print("\nTFLite Models Loaded Successfully!")

# ---- Resolve decoder input indices once (by name, fallback to dtype) ----
dec_input_details = decoder_interpreter.get_input_details()

def _find_tensor(details, name_hint, dtype_fallback):
    """Find tensor index by name hint first, then by dtype."""
    for d in details:
        if name_hint.lower() in d["name"].lower():
            return d["index"]
    # fallback: match by dtype
    for d in details:
        if d["dtype"] == dtype_fallback:
            return d["index"]
    raise ValueError(
        f"Cannot find tensor with name hint '{name_hint}' or dtype {dtype_fallback}. "
        f"Available: {[(d['name'], d['dtype']) for d in details]}"
    )

SEQ_IDX = _find_tensor(dec_input_details, "sequence",  np.int32)
ENC_IDX = _find_tensor(dec_input_details, "encoded",   np.float32)
DEC_OUT_IDX = decoder_interpreter.get_output_details()[0]["index"]

print(f"Decoder → seq tensor index={SEQ_IDX},  enc tensor index={ENC_IDX}")


# ==========================================
# 4. INFERENCE
# ==========================================
def preprocess_image(image_numpy: np.ndarray) -> np.ndarray:
    """
    Replicates training preprocessing exactly:
        decode_and_resize():
            img = tf.image.decode_jpeg(img, channels=3)          # uint8 HWC
            img = tf.image.resize(img, IMAGE_SIZE)               # float32, still [0,255]
            img = tf.image.convert_image_dtype(img, tf.float32)  # maps [0,255] -> [0,1]
    Gradio passes uint8 numpy arrays, so we skip the decode step.
    """
    img = tf.convert_to_tensor(image_numpy, dtype=tf.uint8)          # uint8 [0,255]
    img = tf.image.resize(img, IMAGE_SIZE)                            # float32 [0,255]
    img = tf.image.convert_image_dtype(img, tf.float32)               # float32 [0,1]
    img = tf.expand_dims(img, axis=0)                                  # (1, H, W, 3)
    return img.numpy().astype(np.float32)


def run_encoder(img_np: np.ndarray) -> np.ndarray:
    enc_in_idx  = encoder_interpreter.get_input_details()[0]["index"]
    enc_out_idx = encoder_interpreter.get_output_details()[0]["index"]

    encoder_interpreter.resize_tensor_input(enc_in_idx, img_np.shape)
    encoder_interpreter.allocate_tensors()

    encoder_interpreter.set_tensor(enc_in_idx, img_np)
    encoder_interpreter.invoke()

    out = encoder_interpreter.get_tensor(enc_out_idx)
    return np.array(out, dtype=np.float32)


def run_decoder_step(tokenized_np: np.ndarray, encoded_img: np.ndarray) -> np.ndarray:
    decoder_interpreter.resize_tensor_input(SEQ_IDX, tokenized_np.shape)
    decoder_interpreter.resize_tensor_input(ENC_IDX, encoded_img.shape)
    decoder_interpreter.allocate_tensors()

    decoder_interpreter.set_tensor(SEQ_IDX, tokenized_np)
    decoder_interpreter.set_tensor(ENC_IDX, encoded_img)
    decoder_interpreter.invoke()

    return decoder_interpreter.get_tensor(DEC_OUT_IDX)   # (1, seq_len, vocab_size)


def generate_caption(encoded_img: np.ndarray) -> str:
    decoded_caption = "<start> "

    for i in range(max_decoded_sentence_length):
        tokenized = vectorization([decoded_caption])[:, :-1]
        tokenized_np = np.array(tokenized, dtype=np.int32)

        predictions = run_decoder_step(tokenized_np, encoded_img)

        # predictions shape: (1, SEQ_LENGTH-1, VOCAB_SIZE)
        # At step i, the number of real tokens so far = i+1 (including <start>)
        # We want the prediction AT the last filled position
        # Count non-zero tokens to find correct position
        num_tokens = np.count_nonzero(tokenized_np[0])  # how many tokens filled so far
        pos = max(0, num_tokens - 1)                     # predict next from last position

        if predictions.ndim == 3:
            token_probs = predictions[0, pos, :]
        elif predictions.ndim == 2:
            token_probs = predictions[0, :]
        else:
            token_probs = predictions.flatten()

        # Skip special/reserved tokens (indices 0,1,2,3 = pad, unk, start, end)
        token_probs[:4] = 0

        sampled_token_index = np.argmax(token_probs)
        sampled_token = index_lookup.get(sampled_token_index, "")

        print(f"  step {i:02d} pos={pos} → idx={sampled_token_index}  token='{sampled_token}'")

        if sampled_token in ("<end>", ""):
            break

        decoded_caption += " " + sampled_token

    caption = decoded_caption.replace("<start>", "").strip()
    # Safety strip: remove any leading standalone digits
    caption = re.sub(r'^\d+\s+', '', caption).strip()
    return caption


def process_and_predict(image_numpy):
    if image_numpy is None:
        return "Please upload an image.", None

    try:
        # 1. Preprocess (matches training exactly)
        img_np = preprocess_image(image_numpy)
        print(f"Image → shape={img_np.shape}  min={img_np.min():.4f}  max={img_np.max():.4f}")

        # 2. Encode
        encoded_img = run_encoder(img_np)
        print(f"Encoded → shape={encoded_img.shape}  min={encoded_img.min():.4f}  max={encoded_img.max():.4f}")

        # 3. Decode
        final_caption = generate_caption(encoded_img)
        print(f"Caption: {final_caption}")

        if not final_caption:
            return "Could not generate a caption. Please try another image.", None

        # 4. Text-to-Speech
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            gTTS(text=final_caption, lang="en", slow=False).save(tmp.name)
            return final_caption, tmp.name

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", None


# ==========================================
# 5. GRADIO UI
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 👁️ EchoLens (Edge-Optimized Cloud)")
    gr.Markdown("Real-time captioning and audio descriptions. Optimized with TFLite.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Upload / Capture")
            submit_btn  = gr.Button("Generate Description", variant="primary")
        with gr.Column():
            text_output  = gr.Textbox(label="Caption")
            audio_output = gr.Audio(label="Audio", type="filepath", autoplay=True)

    submit_btn.click(
        fn=process_and_predict,
        inputs=image_input,
        outputs=[text_output, audio_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)