import tensorflow as tf
import keras
from keras import layers
import numpy as np
from scipy.spatial.distance import cosine
import sys
import os
import json
import random
from tqdm import tqdm
from text_embedding import adjust_embedding_dimension, setup_embedding_model, text_to_embedding, embedding_model, embed_dim

fact_patterns = None

@keras.saving.register_keras_serializable(package="wavelet_music")
class TransformerDenoiser(keras.Model):
    def __init__(self, embed_dim, num_heads=4, ff_dim=512):
        super(TransformerDenoiser, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.norm1 = layers.LayerNormalization()
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.norm2 = layers.LayerNormalization()
        self.dropout = layers.Dropout(0.1)
    
    def get_config(self):
        config = super(TransformerDenoiser, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        embed_dim = config.pop('embed_dim')
        num_heads = config.pop('num_heads', 4)
        ff_dim = config.pop('ff_dim', 512)

        model = cls(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)

        for key, value in config.items():
            try:
                setattr(model, key, value)
            except AttributeError:
                pass
                
        return model

    def call(self, inputs):
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=1)

        attn_output = self.attn(inputs, inputs)
        x = self.norm1(inputs + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        x = self.dropout(x)

        if len(inputs.shape) == 3 and inputs.shape[1] == 1:
            x = tf.squeeze(x, axis=1)

        return x

num_samples = 1000
timesteps = 100

def extract_fact_patterns():
    global fact_patterns
    
    if os.path.exists("music_facts.txt"):
        with open("music_facts.txt", "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f.readlines() if line.strip()]

        unique_facts = list(set(texts))

        genre_facts = [t for t in unique_facts if any(x in t.lower() for x in ["jazz", "blues", "rock", "pop", "hip-hop", "reggae", "baroque", "romantic", "renaissance", "classical"])]
        culture_facts = [t for t in unique_facts if any(x in t.lower() for x in ["cultural", "community", "stories", "tradition", "identity"])]
        history_facts = [t for t in unique_facts if any(x in t.lower() for x in ["history", "predates", "ancient", "originated", "began", "emerged"])]
        instrument_facts = [t for t in unique_facts if any(x in t.lower() for x in ["instrument", "flute", "sound"])]
        theory_facts = [t for t in unique_facts if any(x in t.lower() for x in ["philosophers", "ratios", "improvisation", "structured"])]
        
        fact_patterns = {
            "genre": genre_facts,
            "culture": culture_facts,
            "history": history_facts,
            "instrument": instrument_facts,
            "theory": theory_facts,
            "all": unique_facts
        }
        
        return fact_patterns
    else:
        return {"all": []}

def creative_fact_generation(embedding: np.ndarray) -> str:
    global fact_patterns, embedding_model, reference_texts
    
    if fact_patterns is None:
        fact_patterns = extract_fact_patterns()
    
    if embedding_model is None:
        embedding_model = setup_embedding_model()

    similarities = []
    for i, ref_emb in enumerate(reference_embeddings):
        similarity = 1 - cosine(embedding[0], ref_emb)
        similarities.append((similarity, i))

    top_similar = sorted(similarities, reverse=True)[:5]
    top_indices = [idx for _, idx in top_similar]
    top_facts = [reference_texts[idx] for idx in top_indices]

    categories = list(fact_patterns.keys())
    weights = []
    
    for cat in categories[:-1]:
        if not fact_patterns[cat]:
            weights.append(0)
            continue

        cat_raw_embeddings = embedding_model.encode(fact_patterns[cat][:min(10, len(fact_patterns[cat]))])
        cat_embeddings = adjust_embedding_dimension(cat_raw_embeddings, embed_dim)
        avg_cat_emb = np.mean(cat_embeddings, axis=0)

        cat_similarity = 1 - cosine(embedding[0], avg_cat_emb)
        weights.append(max(0, cat_similarity))

    weights = [w + 0.1 * random.random() for w in weights]
    if sum(weights) > 0:
        chosen_cat = random.choices(categories[:-1], weights=weights)[0]
    else:
        chosen_cat = "all"
    
    strategy = random.choice(["direct", "combine", "extend"])
    
    if strategy == "direct" or len(top_facts) < 2:
        return top_facts[0]
    
    elif strategy == "combine" and len(top_facts) >= 2:
        parts1 = top_facts[0].split()
        parts2 = top_facts[1].split()
        
        if len(parts1) > 3 and len(parts2) > 3:
            split_point1 = len(parts1) // 2
            split_point2 = len(parts2) // 2
            new_fact = " ".join(parts1[:split_point1] + parts2[split_point2:])
        else:
            new_fact = top_facts[0]
    
    else:
        base_fact = top_facts[0]
        if chosen_cat != "all" and fact_patterns[chosen_cat]:
            addition = random.choice(fact_patterns[chosen_cat])
            if len(addition.split()) > 4:
                addition_parts = addition.split()
                mid_point = len(addition_parts) // 2
                modifier = " ".join(addition_parts[mid_point:])
                new_fact = base_fact + " " + modifier
            else:
                new_fact = base_fact
        else:
            new_fact = base_fact
    
    return new_fact

def prepare_fine_tuning_data():
    texts = []
    if os.path.exists("music_facts.txt"):
        with open("music_facts.txt", "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f.readlines() if line.strip()]

    global embedding_model
    if embedding_model is None:
        embedding_model = setup_embedding_model()
    embeddings = []
    
    for text in tqdm(texts):
        embedding = text_to_embedding(text)
        embeddings.append(embedding[0])
    embeddings = np.array(embeddings)

    def add_noise(embedding, noise_level=0.1):
        return embedding + noise_level * np.random.randn(*embedding.shape)

    x_train = np.array([add_noise(emb, 0.2) for emb in embeddings]).astype(np.float32)
    y_train = np.array(embeddings).astype(np.float32)

    return x_train, y_train, texts

def train(epochs=100, batch_size=1024):
    x_train, y_train, source_texts = prepare_fine_tuning_data()

    global reference_texts, reference_embeddings, embedding_model
    reference_texts = source_texts

    if embedding_model is None:
        embedding_model = setup_embedding_model()
    
    raw_embeddings = embedding_model.encode(reference_texts)
    reference_embeddings = adjust_embedding_dimension(raw_embeddings, embed_dim)

    extract_fact_patterns()

    model = TransformerDenoiser(embed_dim)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-5),
        loss="mse"
    )

    split_idx = int(len(x_train) * 0.9)
    x_val = x_train[split_idx:]
    y_val = y_train[split_idx:]
    x_train = x_train[:split_idx]
    y_train = y_train[:split_idx]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )

    model.save("music_fact_generator.keras")

    with open('fine_tuning_history.json', 'w') as f:
        history_dict = {}
        for key, values in history.history.items():
            history_dict[key] = [float(v) for v in values]
        json.dump(history_dict, f)
    
    return model

def generate(text_input: str, model_path="music_fact_generator.keras", steps=10) -> str:
    loaded_model = keras.models.load_model(
        model_path, 
        custom_objects={"TransformerDenoiser": TransformerDenoiser}
    )
    input_embedding = text_to_embedding(text_input)
    noise_level = 0.5
    noisy_embedding = input_embedding + noise_level * np.random.randn(*input_embedding.shape)

    current_embedding = noisy_embedding
    for i in range(steps):
        current_embedding = loaded_model.predict(current_embedding, verbose=0)
        if (i+1) % 2 == 0 or i == steps-1:
            creative_fact_generation(current_embedding)
    
    return creative_fact_generation(current_embedding)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py [train|generate] [TEXT(optional)]")
        sys.exit(1)
        
    if sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "generate" and len(sys.argv) > 2:
        text_input = sys.argv[2]
        print(generate(text_input))
    else:
        print("알 수 없는 명령어")