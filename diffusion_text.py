import tensorflow as tf
import keras
from keras import layers
import numpy as np
from scipy.spatial.distance import cosine
import sys
import os
import random
from tqdm import tqdm
from text_embedding import adjust_embedding_dimension, setup_embedding_model, text_to_embedding

@keras.saving.register_keras_serializable(package="wavelet_music")
class TransformerDenoiser(keras.Model):
    def __init__(self, embed_dim=512):
        super(TransformerDenoiser, self).__init__()
        self.embed_dim = embed_dim
        # 인코더 블록
        self.dense1 = layers.Dense(embed_dim * 2, activation='gelu')
        self.dense2 = layers.Dense(embed_dim * 2, activation='gelu')
        
        self.proj1 = layers.Dense(embed_dim, activation=None)
        
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
        
        # 셀프 어텐션 적용 전 차원 축소
        x1_proj = self.proj1(x1)  # 1024 -> 512 차원으로 축소
        
        # 셀프 어텐션 적용 (시퀀스 형태로 변환)
        if len(x1.shape) == 2:  # 배치 크기, 임베딩 차원
            x1_seq = tf.expand_dims(x1_proj, axis=1)  # 배치 크기, 시퀀스 길이(1), 임베딩 차원
        else:
            x1_seq = x1_proj
        
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
        
        # combined 텐서만 반환 (딕셔너리 대신)
        return tf.concat([frequency, volume], axis=-1)
    
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

def find_similar_facts(embedding, reference_texts, reference_embeddings, top_k=3):
    
    if reference_embeddings is None or len(reference_embeddings) == 0:
        return ["No reference embeddings available"]
    
    similarities = []
    for idx, ref_emb in enumerate(reference_embeddings):
        sim = 1 - cosine(embedding.flatten(), ref_emb.flatten())
        similarities.append((sim, idx))
    
    similarities.sort(reverse=True)
    top_idxs = [idx for _, idx in similarities[:top_k]]
    
    return [reference_texts[idx] for idx in top_idxs]

def creative_fact_generation(embedding: np.ndarray, reference_texts, reference_embeddings, fact_patterns) -> str:
    """임베딩을 기반으로 창의적인 음악 사실 생성""" 
    top_facts = find_similar_facts(embedding, reference_texts, reference_embeddings, top_k=3)
    
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
    """파인튜닝 데이터 준비 - 메모리 효율성 개선"""
    texts = []
    if os.path.exists("music_facts.txt"):
        try:
            with open("music_facts.txt", "r", encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        texts.append(line.strip())
            
            print(f"로드된 텍스트 라인 수: {len(texts)}")
        except Exception as e:
            print(f"파일 로딩 중 오류 발생: {e}")
            # 최소한의 샘플 데이터 제공
            texts = ["Sample music fact for testing."]
    
    embedding_model = setup_embedding_model()
    embeddings = []
    
    for text in tqdm(texts):
        embedding = text_to_embedding(text, embedding_model)
        embeddings.append(embedding[0])
    embeddings = np.array(embeddings)

    def add_noise(embedding, noise_level=0.1):
        return embedding + noise_level * np.random.randn(*embedding.shape)

    x_train = np.array([add_noise(emb, 0.2) for emb in embeddings]).astype(np.float32)
    y_train = np.array(embeddings).astype(np.float32)

    return x_train, y_train, texts

def train(epochs=100, batch_size=64, embed_dim=512):
    """트랜스포머 디노이저 모델 학습 - 메모리 효율성 개선"""
    try:
        print("데이터 준비 중...")
        # 데이터를 더 작은 청크로 로드하거나 처리
        x_train, y_train, source_texts = prepare_fine_tuning_data()
        
        # 데이터 크기 체크
        print(f"학습 데이터 크기: {len(x_train)} 샘플")
        if len(x_train) == 0:
            print("경고: 학습 데이터가 없습니다!")
            return None, None, None
        
        # 메모리 사용량 출력 (선택적)
        print(f"x_train 메모리: {x_train.nbytes / (1024 * 1024):.2f} MB")
        print(f"y_train 메모리: {y_train.nbytes / (1024 * 1024):.2f} MB")
        
        actual_dim = x_train.shape[1]
        if actual_dim != embed_dim:
            print(f"경고: 입력 데이터 차원({actual_dim})과 지정된 embed_dim({embed_dim})이 일치하지 않습니다.")
            print(f"embed_dim을 {actual_dim}으로 조정합니다.")
            embed_dim = actual_dim

        # 더 작은 배치 사이즈 및 학습률 조정
        print("모델 컴파일 중...")
        model = TransformerDenoiser(embed_dim)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # 더 낮은 학습률
            loss="mse"
        )
        
        # 메모리 정리
        import gc
        gc.collect()
        
        # 데이터 분할
        split_idx = int(len(x_train) * 0.9)
        x_val = x_train[split_idx:]
        y_val = y_train[split_idx:]
        x_train = x_train[:split_idx]
        y_train = y_train[:split_idx]
        
        # 학습 콜백 설정
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=5, 
                min_lr=1e-6
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath="music_fact_generator_checkpoint",
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        # 메모리에 맞게 학습 수행
        print(f"모델 학습 시작 (배치 크기: {batch_size}, 에폭: {epochs})...")
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("기본 모델 저장 중...")
        model.save("music_fact_generator", save_format='tf')
        
        # 메모리 정리
        del x_train, y_train, x_val, y_val
        gc.collect()
        
        # 오디오 조건부 모델 학습 - 더 단순하게 수정
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
        
        # 보다 효율적인 샘플 수 계산
        sample_size = min(500, len(source_texts))  # 최대 500개 샘플만 사용
        
        # 작은 더미 데이터로 간단하게 학습
        dummy_x = np.random.randn(sample_size, embed_dim).astype(np.float32)
        dummy_targets = np.random.randn(sample_size, 1024 + 128).astype(np.float32)
        
        # 더 작은 배치와 에폭으로 학습
        print("조건부 모델 간단 학습 중...")
        conditioner.fit(
            dummy_x, 
            dummy_targets,
            batch_size=batch_size,
            epochs=100
        )
        
        conditioner.save("audio_conditioner", save_format='tf')
        print("오디오 조건부 모델 저장 완료")
        
        return model, history, conditioner
        
    except tf.errors.ResourceExhaustedError as e:
        print(f"메모리 부족 오류: {e}")
        print("더 작은 배치 크기로 다시 시도해보세요 (예: --batch_size 32)")
        return None, None, None
    except Exception as e:
        print(f"학습 중 오류 발생: {e}")
        return None, None, None

def generate(text_input: str, model_path="music_fact_generator", steps=10, embed_dim=512) -> str:
    """텍스트 생성 기능"""
    loaded_model = keras.models.load_model(
        model_path, 
        custom_objects={"TransformerDenoiser": TransformerDenoiser}
    )
    embedding_model = setup_embedding_model()
    input_embedding = text_to_embedding(text_input, embedding_model)
    noise_level = 0.5
    noisy_embedding = input_embedding + noise_level * np.random.randn(*input_embedding.shape)

    x_train, y_train, source_texts = prepare_fine_tuning_data()
    reference_texts = source_texts
    
    raw_embeddings = embedding_model.encode(reference_texts)
    reference_embeddings = adjust_embedding_dimension(raw_embeddings, embed_dim)
    fact_pattern = extract_fact_patterns()
    
    current_embedding = noisy_embedding
    for i in range(steps):
        current_embedding = loaded_model.predict(current_embedding, verbose=0)
        if (i+1) % 2 == 0 or i == steps-1:
            creative_fact_generation(current_embedding, reference_texts, reference_embeddings, fact_pattern)
    
    return creative_fact_generation(current_embedding, reference_texts, reference_embeddings, fact_pattern)

def generate_audio_seeds(text_input: str, conditioner_path="audio_conditioner", num_seeds=3):
    """텍스트 입력에서 오디오 시드 생성"""
    # 텍스트 임베딩
    empedding_model = setup_embedding_model()
    input_embedding = text_to_embedding(text_input, empedding_model)
    
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
        combined_features = conditioner.predict(noisy_embedding)
        
        # 결과 분할하여 딕셔너리로 변환
        frequency = combined_features[:, :1024]
        volume = combined_features[:, 1024:]
        
        audio_tokens = {
            'frequency': frequency,
            'volume': volume,
            'combined': combined_features
        }
        
        # 결과 저장
        seeds.append(audio_tokens)
    
    return seeds

def combined_text_audio(text_input: str, 
                      text_model_path="music_fact_generator",
                      audio_model_path="text_to_audio_model",
                      diffusion_model_path="diffusion_model",
                      output_path="generated.wav",
                      steps=10):
    """텍스트와 오디오 생성 통합 함수"""
    print(f"입력 텍스트: {text_input}")
    
    # 1. 텍스트 기반 음악 사실 생성
    generated_fact = generate(text_input, model_path=text_model_path, steps=steps)
    print(f"생성된 음악 사실: {generated_fact}")
    
    # 2. 오디오 시드 생성
    seeds = generate_audio_seeds(generated_fact, num_seeds=1)
    
    # 오디오 시드가 None인 경우 처리 추가
    if seeds is None:
        print("오디오 시드 생성 실패")
        return {
            'text': generated_fact,
            'error': "오디오 시드 생성 실패, 조건부 모델을 로드할 수 없습니다."
        }
    return seeds

if __name__ == "__main__":
    reference_texts = []
    reference_embeddings = None
    
    if len(sys.argv) < 2:
        print("Usage: python diffusion_text.py [train|generate] [TEXT(optional)]")
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
    
    # 메모리 제한 설정
    try:
        # TensorFlow 메모리 사용량 제한
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.set_logical_device_configuration(
                physical_devices[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # 4GB 제한
            )
    except Exception as e:
        print(f"메모리 제한 설정 오류: {e}")
    
    # 오류 처리 및 예외 핸들링 강화
    try:
        # 모델 학습 또는 생성
        if sys.argv[1] == 'train':
            try:
                model, history, _ = train(epochs=100, batch_size=256)
                print("모델 학습 완료")
            except Exception as e:
                print(f"학습 중 오류 발생: {e}")
                sys.exit(1)
        elif sys.argv[1] == 'generate':
            if len(sys.argv) < 3:
                print("텍스트 입력이 필요합니다.")
                sys.exit(1)
            else:
                text_input = sys.argv[2]
            
            try:
                output = combined_text_audio(text_input)
            except Exception as e:
                print(f"생성 중 오류 발생: {e}")
                sys.exit(1)
        else:
            print("알 수 없는 명령: " + sys.argv[1])
            print("사용법: python diffusion_text.py [train|generate] [TEXT(optional)]")
    except Exception as e:
        print(f"실행 중 예상치 못한 오류 발생: {e}")
        sys.exit(1)