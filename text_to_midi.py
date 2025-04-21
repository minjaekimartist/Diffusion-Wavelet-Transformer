import glob
import keras
import numpy as np
import os
import sys
from keras import layers
from mido import MidiFile, MidiTrack, Message
from tqdm import tqdm
from text_embedding import text_to_embedding, embed_dim

def midi_to_note_sequence(midi_file_path, length=100):
    mid = MidiFile(midi_file_path)
    note_sequence = []
    for track in mid.tracks:
        time = 0
        for i in range(int(len(track) / length)):
            for j in range(length):
                if i * length + j >= len(track):
                    continue
                msg = track[i * length + j]  
                time += msg.time
                if msg.type == 'note_on':
                    if msg.channel != 9:
                        note_sequence.append([msg.note / 127.0, time / 120.0, 0, msg.velocity / 127.0])
                if msg.type == 'note_off':
                    if msg.channel != 9:
                        for k in range(len(note_sequence)):
                            if note_sequence[len(note_sequence) - 1 - k][0] == msg.note / 127.0:
                                note_sequence[len(note_sequence) - 1 - k][2] = time / 127.0
                                break
        if len(note_sequence) == 0:
            continue
    if len(note_sequence) != 0:
        result = []
        for index in range(int((len(note_sequence) - 1) / length)):
            result.append([note_sequence[index * length: (index + 1) * length]])
        return result
    else:
        return [note_sequence * length]

@keras.utils.register_keras_serializable(package="wavelet_music")
class MidiTransformer(keras.Model):
    def __init__(self, embed_dim, sequence_length, features=4, **kwargs):
        super(MidiTransformer, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length
        self.features = features

        self.dense1 = layers.Dense(256, activation='relu')
        self.dropout1 = layers.Dropout(0.2)
        self.dense2 = layers.Dense(512, activation='relu')
        self.dropout2 = layers.Dropout(0.2)
        self.dense3 = layers.Dense(sequence_length * features, activation='linear')
        self.reshape = layers.Reshape((sequence_length, features))

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        if training:
            x = self.dropout1(x)
        x = self.dense2(x)
        if training:
            x = self.dropout2(x)
        x = self.dense3(x)
        return self.reshape(x)
    
    def get_config(self):
        config = super(MidiTransformer, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "sequence_length": self.sequence_length,
            "features": self.features
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        embed_dim = config.pop('embed_dim', 512)
        sequence_length = config.pop('sequence_length', 100)
        features = config.pop('features', 3)

        return cls(
            embed_dim=embed_dim, 
            sequence_length=sequence_length, 
            features=features, 
            **config
        )

def prepare_midi_data(midi_files, num_files, fixed_length=128):
    midi_sequences = []
    print(f"{min(len(midi_files), num_files)}개의 MIDI 파일 처리 중...")
    
    for midi_file in midi_files[:num_files]:
        seq = midi_to_note_sequence(midi_file, min(len(midi_files), num_files))
        seq = np.array(seq)
        if len(seq) > fixed_length:
            seq = seq[:fixed_length]
        else:
            padding = np.zeros((fixed_length - len(seq), 4))
            seq = np.vstack([seq, padding])
        midi_sequences.extend(seq)
        print(f"{midi_file} 처리 완료.")
    return midi_sequences[:min(len(midi_files), num_files)]

def train(text_data, midi_files, embed_dim=300, batch_size=16, epochs=100, model_checkpoint_path="midi_model.keras"):
    text_data = text_data

    text_embeddings = []
    for text in tqdm(text_data):
        emb = text_to_embedding(text)
        text_embeddings.append(emb[0])
    text_embeddings = np.array(text_embeddings)
    midi_sequences = prepare_midi_data(midi_files, len(text_data))
    print(midi_sequences)
    text_embeddings = text_embeddings
    midi_sequences = midi_sequences

    model = MidiTransformer(embed_dim=embed_dim, sequence_length=len(text_data), features=3)
    
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse')

    model.summary()
    history = model.fit(
        text_embeddings, 
        midi_sequences, 
        batch_size=batch_size, 
        epochs=epochs, 
        verbose=1,
        validation_split=0.1
    )
    model.save(model_checkpoint_path)
    
    return model, history

def generate(text_input, model_path="midi_model.keras"):
    model = keras.models.load_model(
        model_path, 
        custom_objects={"MidiTransformer": MidiTransformer}
    )
    text_embedding = text_to_embedding(text_input)
    text_embedding = text_embedding[0]
    text_embedding = np.expand_dims(text_embedding, axis=0)
    midi_sequence = model.predict(text_embedding)
    midi_sequence = midi_sequence[0]
    return midi_sequence

def save(midi_output, filename='generated_midi.midi', ticks_per_beat=480):
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)

    valid_notes = midi_output[np.logical_and(midi_output[:, 0] > 0, midi_output[:, 2] > 0)]

    valid_notes = valid_notes[np.argsort(valid_notes[:, 1])]

    last_time = 0
    for note_data in valid_notes:
        pitch = int(note_data[0] * 127)
        velocity = int(note_data[3] * 127)
        
        start = int(note_data[1] * 120)
        end = int(note_data[2] * 120)
        if start < last_time:
            time = 0
            index = 0
            while time + track[index].time > start:
                index += 1
            start = (start + last_time) - (time + track[index].time)
            message = Message('note_on', note=pitch, velocity=velocity, time=start)
            track.insert(index + 1, message)
        time = 0
        index = 0
        while time + track[index].time < end + start + last_time:
            time += track[index].time
            index += 1
        message = Message('note_off', note=pitch, velocity=velocity, time=end - time)
        if index == len(track):
            track.append(message)
        else:
            track.insert(index, message)
        last_time = int(note_data[2] * 120)
        print(f"노트: {message}")

    mid.save(filename)
    print(f"MIDI가 {filename}로 저장되었습니다.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python text_to_midi.py [train|generate] [midi_dir/text]")
        sys.exit(1)
        
    if sys.argv[1] == 'train':
        with open("music_facts.txt", "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        midi_files = glob.glob(os.path.join(sys.argv[2], '*.mid'))
        print(f"{len(midi_files)}개의 MIDI 파일을 찾았습니다.")
        train(texts[:1000], midi_files, embed_dim=embed_dim, batch_size=16, epochs=100)
    elif sys.argv[1] == 'generate':
        generated_midi = generate(sys.argv[2])
        save(generated_midi, 'generated_output.midi')
    else:
        print("command not recognized. Use 'train' or 'generate'.")