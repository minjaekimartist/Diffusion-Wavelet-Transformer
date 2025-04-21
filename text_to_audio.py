import glob
import librosa
import numpy as np
import os
import pywt
import scipy.io.wavfile as wav
import sys
import tensorflow as tf
import keras
from keras import callbacks, layers, models
from text_embedding import text_to_embedding
from tqdm import tqdm

def wav_to_wavelet(wav_file_path, wavelet='haar', level=5):
    y, sr = librosa.load(wav_file_path, sr=None)
    coeffs = pywt.wavedec(y, wavelet, level=level)
    coeffs = [c.flatten() for c in coeffs]
    return np.concatenate(coeffs)

@keras.saving.register_keras_serializable(package="wavelet_music")
class AudioTransformer(tf.keras.Model):
    def __init__(self, embed_dim, num_channels=1):
        super(AudioTransformer, self).__init__()
        self.embedding_layer = layers.InputLayer(input_shape=(embed_dim,))
        self.transformer = layers.MultiHeadAttention(num_heads=4, key_dim=embed_dim)
        self.conv1 = layers.Conv1D(128, 5, activation='relu', padding='same')
        self.conv2 = layers.Conv1D(64, 5, activation='relu', padding='same')
        self.output_layer = layers.Conv1D(num_channels, 5, activation='tanh', padding='same')

    def call(self, inputs):
        x = self.embedding_layer(inputs)
        x = tf.expand_dims(x, axis=1)
        x = self.transformer(x, x)
        x = self.conv1(x)
        x = self.conv2(x)
        output = self.output_layer(x)
        return output

def prepare_audio_data(wav_files, maxlen=500):
    audio_features = [wav_to_wavelet(wav_file) for wav_file in wav_files]
    audio_features = [feature[:maxlen] for feature in audio_features]
    return np.array(audio_features)

def train(text_data, wav_files, embed_dim=512, batch_size=32, epochs=100, model_checkpoint_path="audio_model.keras"):
    text_embeddings = np.array([text_to_embedding(text) for text in tqdm(text_data)])
    audio_features = prepare_audio_data(wav_files)
    model = AudioTransformer(embed_dim, num_channels=1)
    model.compile(optimizer='adam', loss='mean_squared_error')

    checkpoint_callback = callbacks.ModelCheckpoint(model_checkpoint_path, save_best_only=True)
    model.fit(text_embeddings, audio_features, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint_callback])
    model.save(model_checkpoint_path)
    print(f"Text-to-Audio model trained and saved at {model_checkpoint_path}")
    return model

def generate_audio(text_input, model, embed_dim):
    loaded_model = models.load_model(
        model, 
        custom_objects={"MidiTransformer": AudioTransformer}
    )
    text_embedding = text_to_embedding(text_input, embed_dim)
    text_embedding = np.expand_dims(text_embedding, axis=0)
    pcm_audio = loaded_model.predict(text_embedding)
    pcm_audio = np.int16(pcm_audio * 32767)
    return pcm_audio

def save(pcm_audio, sample_rate, filename='generated_audio.wav'):
    wav.write(filename, sample_rate, pcm_audio)
    print(f"Generated audio saved as {filename}")

if __name__ == "__main__":
    if sys.argv[1] == 'train':
        with open("music_facts.txt", "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f.readlines() if line.strip()]
        audio_files = glob.glob(os.path.join(sys.argv[2], '*.wav'))
        model = train(texts, audio_files, embed_dim=300, batch_size=32, epochs=100)
    if sys.argv[1] == 'generate':
        audio = generate_audio(sys.argv[2], model, embed_dim=300)
        save(audio, 'generated_output.wav')
