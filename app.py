"""
EchoLens: Edge-Optimized Image Captioning Pipeline
--------------------------------------------------
This script performs inference using quantized TFLite models.
It handles pre-processing, autoregressive decoding, and text-to-speech.
Optimized for deployment on Hugging Face Docker Spaces.
"""

import os
import re
import numpy as np
import tensorflow as tf
from keras.layers import TextVectorization
import pickle
import yaml
import gradio as gr
from gtts import gTTS

# Ensure TensorFlow is used as the Keras backend
os.environ["KERAS_BACKEND"] = "tensorflow"

# ==========================================
# 1. INITIALIZATION & CONFIGURATION
# ==========================================
# Load configuration from yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

IMAGE_SIZE = tuple(config["image_preprocessing"]["IMAGE_SIZE"])
SEQ_LENGTH = config["training"]["SEQ_LENGTH"]
VOCAB_SIZE = config["training"]["VOCAB_SIZE"]
VOCAB_PATH = config["paths"]["vocab_file"]

# Text standardization logic (matching training phase)
strip_chars = r"!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~".replace("<", "").replace(">", "")
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

# Load vocabulary and setup vectorization
vectorization = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQ_LENGTH,
    standardize=custom_standardization,
)

with open(VOCAB_PATH, "rb") as f:
    vectorization_loaded_data = pickle.load(f)
vectorization.set_vocabulary(vectorization_loaded_data['vocab'])
index_lookup = dict(zip(range(len(vectorization_loaded_data['vocab'])), vectorization_loaded_data['vocab']))
max_decoded_sentence_length = SEQ_LENGTH - 1

# ==========================================
# 2. LOAD TFLITE INTERPRETERS
# ==========================================
print("Loading Lightweight TFLite Models...")

# Initialize Interpreters
encoder_interpreter = tf.lite.Interpreter(model_path="echolens_encoder_quantized.tflite")
encoder_interpreter.allocate_tensors()

decoder_interpreter = tf.lite.Interpreter(model_path="echolens_decoder_quantized.tflite")
decoder_interpreter.allocate_tensors()

print("TFLite Models Loaded Successfully!")

# ==========================================
# 3. GRADIO INFERENCE PIPELINE (Robust Implementation)
# ==========================================
import tempfile

def process_and_predict(image_numpy):
    if image_numpy is None:
        return "Please upload an image.", None
    
    # 1. Preprocess
    img = tf.convert_to_tensor(image_numpy)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, 0)
    
    # 2. Run Encoder
    enc_in = encoder_interpreter.get_input_details()[0]['index']
    enc_out = encoder_interpreter.get_output_details()[0]['index']
    encoder_interpreter.set_tensor(enc_in, img)
    encoder_interpreter.invoke()
    encoded_img = encoder_interpreter.get_tensor(enc_out)
    
    # 3. Prepare Decoder
    dec_details = decoder_interpreter.get_input_details()
    seq_idx = next(d['index'] for d in dec_details if 'sequence' in d['name'].lower())
    enc_idx = next(d['index'] for d in dec_details if 'encoded' in d['name'].lower())
    dec_out_idx = decoder_interpreter.get_output_details()[0]['index']

    # 4. Decoding Loop
    decoded_caption = "<start> "
    for i in range(max_decoded_sentence_length):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        tokenized_caption = tf.cast(tokenized_caption, tf.int32)
        
        # FIX: Only allocate memory on the first iteration
        if i == 0:
            decoder_interpreter.resize_tensor_input(seq_idx, tokenized_caption.shape)
            decoder_interpreter.resize_tensor_input(enc_idx, encoded_img.shape)
            decoder_interpreter.allocate_tensors()
        
        decoder_interpreter.set_tensor(seq_idx, tokenized_caption.numpy())
        decoder_interpreter.set_tensor(enc_idx, encoded_img.numpy())
        decoder_interpreter.invoke()
        
        predictions = decoder_interpreter.get_tensor(dec_out_idx)
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup.get(sampled_token_index, "<unk>")
        
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token

    final_caption = decoded_caption.replace("<start> ", "").replace(" <end>", "").strip()
    
    # Use tempfile for writing to ensure compatibility with Docker restricted FS
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        tts = gTTS(text=final_caption, lang='en', slow=False)
        tts.save(temp_audio.name)
        return final_caption, temp_audio.name

# ==========================================
# 4. GRADIO INTERFACE
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 👁️ EchoLens (Edge-Optimized Cloud)")
    gr.Markdown("Real-time captioning and audio descriptions. Optimized with TFLite.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Upload/Capture")
            submit_btn = gr.Button("Generate Description", variant="primary")
        with gr.Column():
            text_output = gr.Textbox(label="Caption")
            audio_output = gr.Audio(label="Audio", type="filepath", autoplay=True)
            
    submit_btn.click(process_and_predict, inputs=image_input, outputs=[text_output, audio_output])

if __name__ == "__main__":
    # Required for Docker Spaces
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)