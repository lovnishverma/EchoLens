import os
import re
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.applications import efficientnet
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
MODEL_PATH = config["paths"]["model_path"]
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
# 2. MODEL ARCHITECTURE STUBS
# ==========================================
def get_cnn_model():
    base_model = efficientnet.EfficientNetB0(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet")
    base_model.trainable = False
    base_model_out = layers.Reshape((-1, base_model.output.shape[-1]))(base_model.output)
    return keras.models.Model(base_model.input, base_model_out)

@keras.utils.register_keras_serializable()
class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.0)
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.dense_1 = layers.Dense(embed_dim, activation="relu")
    def call(self, inputs, training=False, mask=None):
        inputs = self.layernorm_1(inputs)
        inputs = self.dense_1(inputs)
        attention_output_1 = self.attention_1(query=inputs, value=inputs, key=inputs, training=training)
        return self.layernorm_2(inputs + attention_output_1)

@keras.utils.register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)
        self.embed_scale = tf.math.sqrt(tf.cast(embed_dim, tf.float32))
    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs) * self.embed_scale
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions
    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

@keras.utils.register_keras_serializable()
class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.1)
        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.1)
        self.ffn_layer_1 = layers.Dense(ff_dim, activation="relu")
        self.ffn_layer_2 = layers.Dense(embed_dim)
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.embedding = PositionalEmbedding(embed_dim=embed_dim, sequence_length=SEQ_LENGTH, vocab_size=VOCAB_SIZE)
        self.out = layers.Dense(VOCAB_SIZE, activation="softmax")
        self.dropout_1 = layers.Dropout(0.3)
        self.dropout_2 = layers.Dropout(0.5)
        self.supports_masking = True
    def call(self, inputs, encoder_outputs, training=False, mask=None):
        inputs = self.embedding(inputs)
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)
        attention_output_1 = self.attention_1(query=inputs, value=inputs, key=inputs, attention_mask=combined_mask, training=training)
        out_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(query=out_1, value=encoder_outputs, key=encoder_outputs, attention_mask=padding_mask, training=training)
        out_2 = self.layernorm_2(out_1 + attention_output_2)
        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)
        ffn_out = self.layernorm_3(ffn_out + out_2, training=training)
        ffn_out = self.dropout_2(ffn_out, training=training)
        return self.out(ffn_out)
    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat([tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], axis=0)
        return tf.tile(mask, mult)

@keras.utils.register_keras_serializable()
class ImageCaptioningModel(keras.Model):
    def __init__(self, cnn_model, encoder, decoder, num_captions_per_image=5, image_aug=None, **kwargs):
        super().__init__(**kwargs)
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.num_captions_per_image = num_captions_per_image
        self.image_aug = image_aug
    def get_config(self):
        config = super().get_config()
        config.update({
            "cnn_model": keras.saving.serialize_keras_object(self.cnn_model),
            "encoder": keras.saving.serialize_keras_object(self.encoder),
            "decoder": keras.saving.serialize_keras_object(self.decoder),
            "num_captions_per_image": self.num_captions_per_image,
            "image_aug": keras.saving.serialize_keras_object(self.image_aug),
        })
        return config
    @classmethod
    def from_config(cls, config):
        cnn_model = keras.saving.deserialize_keras_object(config.pop("cnn_model"))
        encoder = keras.saving.deserialize_keras_object(config.pop("encoder"))
        decoder = keras.saving.deserialize_keras_object(config.pop("decoder"))
        image_aug = keras.saving.deserialize_keras_object(config.pop("image_aug"))
        num_captions_per_image = config.pop("num_captions_per_image")
        return cls(cnn_model=cnn_model, encoder=encoder, decoder=decoder, num_captions_per_image=num_captions_per_image, image_aug=image_aug, **config)

# Load the trained Keras model
print("Loading Model...")
caption_model = keras.models.load_model(MODEL_PATH)
caption_model.build(input_shape=(1, *IMAGE_SIZE, 3))
print("Model Loaded Successfully!")

# ==========================================
# 3. GRADIO INFERENCE PIPELINE
# ==========================================
def process_and_predict(image_numpy):
    if image_numpy is None:
        return "Please upload an image.", None
    
    # Preprocess Gradio Image (Numpy Array)
    img = tf.convert_to_tensor(image_numpy)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, 0)
    
    # Extract Features
    img_features = caption_model.cnn_model(img)
    encoded_img = caption_model.encoder(img_features, training=False)
    
    # Autoregressive Decoder
    decoded_caption = "<start> "
    for i in range(max_decoded_sentence_length):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(tokenized_caption, encoded_img, training=False, mask=mask)
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup.get(sampled_token_index, "<unk>")
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token

    final_caption = decoded_caption.replace("<start> ", "").replace(" <end>", "").strip()
    
    # Generate Audio using gTTS
    audio_path = "output_caption.mp3"
    tts = gTTS(text=final_caption, lang='en', slow=False)
    tts.save(audio_path)
    
    return final_caption, audio_path

# ==========================================
# 4. GRADIO INTERFACE SETUP
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 👁️ EchoLens Cloud")
    gr.Markdown("Upload an image or use your webcam to generate a real-time text caption and audio description.")
    
    with gr.Row():
        with gr.Column():
            # Input supports both file upload and webcam
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
    # Server name and port are REQUIRED for Hugging Face Docker Spaces
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)