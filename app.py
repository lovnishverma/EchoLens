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
import tempfile

os.environ["KERAS_BACKEND"] = "tensorflow"

# ==========================================
# 1. INITIALIZATION & CONFIGURATION
# ==========================================
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

IMAGE_SIZE = tuple(config["image_preprocessing"]["IMAGE_SIZE"])
SEQ_LENGTH = config["training"]["SEQ_LENGTH"]
VOCAB_SIZE = config["training"]["VOCAB_SIZE"]
VOCAB_PATH = config["paths"]["vocab_file"]

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

vectorization.set_vocabulary(vectorization_loaded_data['vocab'])
index_lookup = dict(zip(range(len(vectorization_loaded_data['vocab'])), vectorization_loaded_data['vocab']))
max_decoded_sentence_length = SEQ_LENGTH - 1

# ==========================================
# 2. LOAD TFLITE INTERPRETERS
# ==========================================
print("Loading Lightweight TFLite Models...")

encoder_interpreter = tf.lite.Interpreter(model_path="echolens_encoder_quantized.tflite")
encoder_interpreter.allocate_tensors()

decoder_interpreter = tf.lite.Interpreter(model_path="echolens_decoder_quantized.tflite")
decoder_interpreter.allocate_tensors()

print("TFLite Models Loaded Successfully!")

# ==========================================
# 3. INFERENCE PIPELINE
# ==========================================
def process_and_predict(image_numpy):
    if image_numpy is None:
        return "Please upload an image.", None

    try:
        # 1. Preprocess image
        img = tf.convert_to_tensor(image_numpy, dtype=tf.float32)
        img = tf.image.resize(img, IMAGE_SIZE)
        img = img / 255.0  # normalize to [0, 1]
        img = tf.expand_dims(img, 0)
        img_np = np.array(img, dtype=np.float32)

        # 2. Run Encoder
        enc_input_details = encoder_interpreter.get_input_details()
        enc_output_details = encoder_interpreter.get_output_details()

        encoder_interpreter.resize_tensor_input(enc_input_details[0]['index'], img_np.shape)
        encoder_interpreter.allocate_tensors()
        encoder_interpreter.set_tensor(enc_input_details[0]['index'], img_np)
        encoder_interpreter.invoke()
        encoded_img = encoder_interpreter.get_tensor(enc_output_details[0]['index'])  # already np.ndarray
        encoded_img = np.array(encoded_img, dtype=np.float32)

        # 3. Identify decoder inputs by dtype (int32=tokens, float32=image features)
        dec_details = decoder_interpreter.get_input_details()
        for d in dec_details:
            print(f"  Decoder input → name: {d['name']}, shape: {d['shape']}, dtype: {d['dtype']}")

        int32_inputs  = [d for d in dec_details if d['dtype'] == np.int32]
        float32_inputs = [d for d in dec_details if d['dtype'] == np.float32]

        if not int32_inputs or not float32_inputs:
            raise ValueError(
                f"Could not identify decoder inputs by dtype. "
                f"Found dtypes: {[d['dtype'] for d in dec_details]}. "
                f"Check Logs for tensor names."
            )

        seq_idx = int32_inputs[0]['index']
        enc_idx = float32_inputs[0]['index']
        dec_out_idx = decoder_interpreter.get_output_details()[0]['index']

        # 4. Autoregressive decoding loop
        decoded_caption = "<start> "

        for i in range(max_decoded_sentence_length):
            tokenized_caption = vectorization([decoded_caption])[:, :-1]
            tokenized_np = np.array(tokenized_caption, dtype=np.int32)

            # Must resize + reallocate every iteration after first tensor shape is set
            decoder_interpreter.resize_tensor_input(seq_idx, tokenized_np.shape)
            decoder_interpreter.resize_tensor_input(enc_idx, encoded_img.shape)
            decoder_interpreter.allocate_tensors()

            # Use np.array() — safe for both ndarray and EagerTensor
            decoder_interpreter.set_tensor(seq_idx, tokenized_np)
            decoder_interpreter.set_tensor(enc_idx, encoded_img)
            decoder_interpreter.invoke()

            predictions = decoder_interpreter.get_tensor(dec_out_idx)
            sampled_token_index = np.argmax(predictions[0, i, :])
            sampled_token = index_lookup.get(sampled_token_index, "")

            if sampled_token in ("<end>", ""):
                break

            decoded_caption += " " + sampled_token

        final_caption = decoded_caption.replace("<start>", "").strip()
        print(f"Generated caption: {final_caption}")

        if not final_caption:
            return "Could not generate a caption. Please try another image.", None

        # 5. Text-to-Speech
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts = gTTS(text=final_caption, lang='en', slow=False)
            tts.save(temp_audio.name)
            return final_caption, temp_audio.name

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", None


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

    submit_btn.click(
        fn=process_and_predict,
        inputs=image_input,
        outputs=[text_output, audio_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)