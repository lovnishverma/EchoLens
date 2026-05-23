import os
import re
import numpy as np
import tensorflow as tf
from keras.layers import TextVectorization
import pickle
import yaml
import gradio as gr
from gtts import gTTS

os.environ["KERAS_BACKEND"] = "tensorflow"

# ==========================================
# 1. LOAD CONFIG & VOCABULARY
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
vocab = vectorization_loaded_data['vocab']
vectorization.set_vocabulary(vocab)
index_lookup = dict(zip(range(len(vocab)), vocab))
max_decoded_sentence_length = SEQ_LENGTH - 1

# ==========================================
# 2. LOAD TFLITE MODELS
# ==========================================
print("Loading Lightweight TFLite Models...")

# Initialize Encoder Interpreter
encoder_interpreter = tf.lite.Interpreter(model_path="echolens_encoder_quantized.tflite")
encoder_interpreter.allocate_tensors()

# Initialize Decoder Interpreter
decoder_interpreter = tf.lite.Interpreter(model_path="echolens_decoder_quantized.tflite")
decoder_interpreter.allocate_tensors()

print("Models Loaded Successfully!")

# ==========================================
# 3. GRADIO INFERENCE PIPELINE
# ==========================================
def process_and_predict(image_numpy):
    if image_numpy is None:
        return "Please upload an image.", None
    
    # 1. Preprocess Gradio Image (Numpy Array to Tensor)
    img = tf.convert_to_tensor(image_numpy)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, 0)
    
    # 2. Run TFLite Encoder
    encoder_input_idx = encoder_interpreter.get_input_details()[0]['index']
    encoder_output_idx = encoder_interpreter.get_output_details()[0]['index']
    
    encoder_interpreter.set_tensor(encoder_input_idx, img)
    encoder_interpreter.invoke()
    encoded_img = encoder_interpreter.get_tensor(encoder_output_idx)
    
    # 3. Run TFLite Decoder (Autoregressive Loop)
    input_details = decoder_interpreter.get_input_details()
    seq_input_idx = next(d['index'] for d in input_details if 'sequence' in d['name'].lower())
    enc_input_idx = next(d['index'] for d in input_details if 'encoded' in d['name'].lower())
    decoder_output_idx = decoder_interpreter.get_output_details()[0]['index']

    decoded_caption = "<start> "
    for i in range(max_decoded_sentence_length):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        
        # Resize decoder input tensors dynamically based on the growing sequence length
        decoder_interpreter.resize_tensor_input(seq_input_idx, tokenized_caption.shape)
        decoder_interpreter.resize_tensor_input(enc_input_idx, encoded_img.shape)
        decoder_interpreter.allocate_tensors()
        
        # Set tensors and invoke
        decoder_interpreter.set_tensor(seq_input_idx, tokenized_caption)
        decoder_interpreter.set_tensor(enc_input_idx, encoded_img)
        decoder_interpreter.invoke()
        
        # Extract prediction
        predictions = decoder_interpreter.get_tensor(decoder_output_idx)
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup.get(sampled_token_index, "<unk>")
        
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token

    final_caption = decoded_caption.replace("<start> ", "").replace(" <end>", "").strip()
    
    # 4. Generate Audio using gTTS
    audio_path = "output_caption.mp3"
    tts = gTTS(text=final_caption, lang='en', slow=False)
    tts.save(audio_path)
    
    return final_caption, audio_path

# ==========================================
# 4. GRADIO INTERFACE SETUP
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 👁️ EchoLens (Edge-Optimized)")
    gr.Markdown("Upload an image or use your webcam to generate a real-time text caption and audio description. *Powered by TFLite.*")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Upload Image or Capture via Webcam")
            submit_btn = gr.Button("Generate Description", variant="primary")
            
        with gr.Column():
            text_output = gr.Textbox(label="Predicted Caption", lines=3)
            audio_output = gr.Audio(label="Audio Description", type="filepath", autoplay=True)
            
    submit_btn.click(
        fn=process_and_predict,
        inputs=image_input,
        outputs=[text_output, audio_output]
    )

if __name__ == "__main__":
    # REQUIRED FOR DOCKER
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)