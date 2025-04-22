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
from text_embedding import adjust_embedding_dimension, setup_embedding_model, text_to_embedding, embedding_model, embed_dim, text_to_tokens

fact_patterns = None

@keras.saving.register_keras_serializable(package="wavelet_music")
class TransformerDenoiser(keras.Model):
    def __init__(self, embed_dim=512):
        super(TransformerDenoiser, self).__init__()
        
        # 인코더 블록
        self.dense1 = layers.Dense(embed_dim * 2, activation='gelu')
        self.dense2 = layers.Dense(embed_dim * 2, activation='gelu')
        
        # 어텐션 메커니즘
        self.attention = layers.MultiHeadAttention(
            num_heads=8, 
            key_dim=embed_dim // 8
        )
        
        # 디코더 블록
        self.dense3 = layers.Dense(embed_dim * 2, activation='gelu')
        self.dense4 = layers.Dense(embed_dim, activation=None)
        
        # 정규화 레이어
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # 드롭아웃 레이어
        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)
        
    def call(self, inputs, training=False):
        # 입력이 배치형태인지 확인
        if len(inputs.shape) == 1:
            inputs = tf.expand_dims(inputs, axis=0)
        
        # 입력 정규화
        x = self.norm1(inputs)
        
        # 인코더 경로
        x1 = self.dense1(x)
        x1 = self.dropout1(x1, training=training)
        x1 = self.dense2(x1)
        
        # 셀프 어텐션 적용 (시퀀스 형태로 변환)
        if len(x1.shape) == 2:  # 배치 크기, 임베딩 차원
            x1_seq = tf.expand_dims(x1, axis=1)  # 배치 크기, 시퀀스 길이(1), 임베딩 차원
        else:
            x1_seq = x1
        
        attention_output = self.attention(x1_seq, x1_seq, x1_seq, training=training)
        
        # 시퀀스 차원 제거 (형태 복원)
        if len(attention_output.shape) == 3 and attention_output.shape[1] == 1:
            attention_output = tf.squeeze(attention_output, axis=1)
        
        # 첫 번째 스킵 연결
        x2 = x + attention_output
        x2 = self.norm2(x2)
        
        # 디코더 경로
        x3 = self.dense3(x2)
        x3 = self.dropout2(x3, training=training)
        x3 = self.dense4(x3)
        
        # 두 번째 스킵 연결
        return x + x3

@keras.saving.register_keras_serializable(package="wavelet_music")
class AudioDiffusionConditioner(keras.Model):
    """텍스트 임베딩을 오디오 디퓨전 모델의 입력으로 변환하는 조건부 모델"""
    
    def __init__(self, text_embed_dim=512, audio_token_dim=256, frequency_dim=1024, volume_dim=128):
        super(AudioDiffusionConditioner, self).__init__()
        
        # 텍스트 입력과 오디오 출력 차원
        self.text_embed_dim = text_embed_dim
        self.audio_token_dim = audio_token_dim
        self.frequency_dim = frequency_dim
        self.volume_dim = volume_dim
        
        # 트랜스포머 기반 텍스트-오디오 변환
        self.text_encoder = keras.Sequential([
            layers.InputLayer(input_shape=(text_embed_dim,)),
            layers.Dense(512, activation='gelu'),
            layers.Dropout(0.1),
            layers.Dense(768, activation='gelu'),
            layers.Dropout(0.1),
            layers.Dense(audio_token_dim, activation='linear')
        ])
        
        # 멀티헤드 셀프 어텐션
        self.attention = layers.MultiHeadAttention(
            num_heads=8, 
            key_dim=audio_token_dim // 8
        )
        
        # 주파수 및 볼륨 프로젝션
        self.frequency_proj = layers.Dense(frequency_dim, activation='tanh')
        self.volume_proj = layers.Dense(volume_dim, activation='sigmoid')
        
        # 정규화
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, inputs, training=False):
        # 텍스트 인코딩
        text_features = self.text_encoder(inputs)
        
        # 어텐션을 위한 시퀀스 차원 추가
        text_seq = tf.expand_dims(text_features, axis=1)
        
        # 셀프 어텐션
        attended = self.attention(text_seq, text_seq, text_seq, training=training)
        attended = tf.squeeze(attended, axis=1)  # 시퀀스 차원 제거
        
        # 스킵 연결 및 정규화
        features = self.layernorm(text_features + attended)
        
        # 오디오 특성 생성
        frequency = self.frequency_proj(features)
        volume = self.volume_proj(features)
        
        # 오디오 토큰 출력
        return {
            'frequency': frequency,
            'volume': volume,
            'combined': tf.concat([frequency, volume], axis=-1)
        }
    
    def get_config(self):
        config = super(AudioDiffusionConditioner, self).get_config()
        config.update({
            'text_embed_dim': self.text_embed_dim,
            'audio_token_dim': self.audio_token_dim,
            'frequency_dim': self.frequency_dim,
            'volume_dim': self.volume_dim,
        })
        return config

# 디퓨전 모델 설정
num_samples = 1000
timesteps = 100

def extract_fact_patterns():
    global fact_patterns
    
    fact_patterns = {
        "genre": [
            "is considered a pioneering work in the genre",
            "revolutionized the genre",
            "established many of the conventions of the genre",
            "is one of the most influential examples of the genre",
            "is widely regarded as defining the genre"
        ],
        "impact": [
            "has had a profound influence on",
            "has been cited as a major inspiration by",
            "has shaped generations of",
            "continues to influence contemporary",
            "is considered essential listening for aspiring"
        ],
        "composition": [
            "features innovative use of",
            "is known for its complex arrangements of",
            "showcases virtuosic performances on",
            "incorporates unconventional rhythms and",
            "blends traditional elements with innovative"
        ],
        "music_theory": [
            "employs polyrhythmic structures",
            "uses modal interchange",
            "features harmonically rich chord progressions",
            "incorporates microtonal elements",
            "demonstrates mastery of dynamic contrast"
        ],
        "all": []
    }
    
    for cat in fact_patterns:
        if cat != "all":
            fact_patterns["all"].extend(fact_patterns[cat])
    
    return fact_patterns

def find_similar_facts(embedding, top_k=3):
    global reference_texts, reference_embeddings
    
    if reference_embeddings is None or len(reference_embeddings) == 0:
        return ["No reference embeddings available"]
    
    similarities = []
    for idx, ref_emb in enumerate(reference_embeddings):
        sim = 1 - cosine(embedding.flatten(), ref_emb.flatten())
        similarities.append((sim, idx))
    
    similarities.sort(reverse=True)
    top_idxs = [idx for _, idx in similarities[:top_k]]
    
    return [reference_texts[idx] for idx in top_idxs]

def creative_fact_generation(embedding: np.ndarray) -> str:
    """임베딩을 기반으로 창의적인 음악 사실 생성"""
    global fact_patterns
    
    if fact_patterns is None:
        extract_fact_patterns()
    
    top_facts = find_similar_facts(embedding, top_k=3)
    
    # 유사 사실이 없는 경우 대비
    if not top_facts or top_facts[0] == "No reference embeddings available":
        return "This composition creates an innovative sound palette using advanced wavelet techniques."
    
    # 랜덤 카테고리 선택
    categories = list(fact_patterns.keys())
    chosen_cat = random.choice(categories)
    
    # 두 사실을 조합하거나 패턴 적용
    if len(top_facts) > 1 and random.random() < 0.5:
        # 두 사실 결합
        parts1 = top_facts[0].split()
        parts2 = top_facts[1].split()
        
        if len(parts1) > 4 and len(parts2) > 4:
            split_point1 = random.randint(2, len(parts1) - 2)
            split_point2 = random.randint(2, len(parts2) - 2)
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
    """파인튜닝 데이터 준비"""
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
    """트랜스포머 디노이저 모델 학습"""
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
    
    # 오디오 조건부 모델 학습 (선택적)
    print("오디오 조건부 모델 학습 시작...")
    conditioner = AudioDiffusionConditioner(
        text_embed_dim=embed_dim,
        audio_token_dim=256,
        frequency_dim=1024,
        volume_dim=128
    )
    
    conditioner.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="mse"
    )
    
    # 임의의 타겟 데이터 생성 (실제로는 오디오 특성이 필요)
    dummy_targets = {
        'frequency': np.random.randn(len(x_train), 1024).astype(np.float32),
        'volume': np.random.rand(len(x_train), 128).astype(np.float32),
        'combined': np.random.randn(len(x_train), 1024 + 128).astype(np.float32)
    }
    
    # 조건부 모델 간단 학습 (실제 구현 시 적절한 데이터 필요)
    conditioner.fit(
        x_train, 
        dummy_targets['combined'],
        batch_size=batch_size,
        epochs=5  # 짧게 학습
    )
    
    conditioner.save("audio_conditioner.keras")
    print("오디오 조건부 모델 저장됨: audio_conditioner.keras")
    
    return model, history, conditioner

def generate(text_input: str, model_path="music_fact_generator.keras", steps=10) -> str:
    """텍스트 생성 기능"""
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

def generate_audio_seeds(text_input: str, conditioner_path="audio_conditioner.keras", num_seeds=3):
    """텍스트 입력에서 오디오 시드 생성"""
    # 텍스트 임베딩
    input_embedding = text_to_embedding(text_input)
    
    # 조건부 모델 로드
    try:
        conditioner = keras.models.load_model(
            conditioner_path,
            custom_objects={"AudioDiffusionConditioner": AudioDiffusionConditioner}
        )
    except Exception as e:
        print(f"조건부 모델 로드 오류: {e}")
        return None
    
    # 오디오 특성 생성
    seeds = []
    for i in range(num_seeds):
        # 약간의 랜덤성 추가
        noise = np.random.randn(*input_embedding.shape) * 0.05
        noisy_embedding = input_embedding + noise
        
        # 조건부 특성 생성
        audio_tokens = conditioner.predict(noisy_embedding)
        
        # 결과 저장
        seeds.append(audio_tokens)
    
    return seeds

def combined_text_audio(text_input: str, 
                      text_model_path="music_fact_generator.keras",
                      audio_model_path="text_to_audio_model.keras",
                      diffusion_model_path="diffusion_model.keras",
                      output_path="generated.wav",
                      steps=10):
    """텍스트와 오디오 생성 통합 함수"""
    print(f"입력 텍스트: {text_input}")
    
    # 1. 텍스트 기반 음악 사실 생성
    generated_fact = generate(text_input, model_path=text_model_path, steps=steps)
    print(f"생성된 음악 사실: {generated_fact}")
    
    # 2. 오디오 시드 생성
    seeds = generate_audio_seeds(generated_fact, num_seeds=1)
    
    # 3. 텍스트-오디오 모델로 오디오 생성
    from text_to_audio import generate_audio_from_text
    
    try:
        # 텍스트에서 오디오 생성
        audio_samples = generate_audio_from_text(
            generated_fact,
            tokenizer_model_path=audio_model_path,
            diffusion_model_path=diffusion_model_path,
            output_path=output_path
        )
        
        print(f"오디오 생성 완료: {output_path}")
        return {
            'text': generated_fact,
            'audio_path': output_path,
            'audio_samples': audio_samples
        }
    except Exception as e:
        print(f"오디오 생성 오류: {e}")
        return {
            'text': generated_fact,
            'error': str(e)
        }

if __name__ == "__main__":
    reference_texts = []
    reference_embeddings = None
    
    if len(sys.argv) < 2:
        print("Usage: python script.py [train|generate] [TEXT(optional)]")
        sys.exit(1)
    
    # 사용 가능한 장치 확인
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU 사용 가능: {len(gpus)}개")
        except Exception as e:
            print(f"GPU 설정 오류: {e}")
    
    # 모델 학습 또는 생성
    if sys.argv[1] == 'train':
        model, history, _ = train(epochs=100, batch_size=32)
        print("모델 학습 완료")
    elif sys.argv[1] == 'generate':
        if len(sys.argv) < 3:
            text_input = "electronic music with complex rhythms"
        else:
            text_input = sys.argv[2]
            
        output = combined_text_audio(text_input)
        print(f"생성 결과: {output['text']}")
    else:
        print("알 수 없는 명령: " + sys.argv[1])
        print("사용법: python script.py [train|generate] [TEXT(optional)]")