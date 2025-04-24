import numpy as np
import tensorflow as tf
import keras
from keras import layers
import json
from tqdm import tqdm
from text_embedding import text_to_embedding, adjust_embedding_dimension
from main import Data

class TextToAudioTokenizer(keras.Model):
    """텍스트를 오디오 토큰으로 변환하는 트랜스포머 모델"""
    
    def __init__(self, 
                 text_embed_dim=512, 
                 audio_token_dim=256,
                 frequency_dim=1024,
                 volume_dim=128,
                 time_dim=2048,
                 num_heads=8,
                 num_layers=6,
                 dropout_rate=0.1,
                 **kwargs):
        super(TextToAudioTokenizer, self).__init__(**kwargs)
        
        self.text_embed_dim = text_embed_dim
        self.audio_token_dim = audio_token_dim
        self.frequency_dim = frequency_dim
        self.volume_dim = volume_dim
        self.time_dim = time_dim
        
        # 텍스트 임베딩 레이어 (고정 차원의 텍스트 입력)
        self.text_input_layer = layers.InputLayer(input_shape=(text_embed_dim,))
        self.text_dense = layers.Dense(audio_token_dim, activation='relu')
        
        # 텍스트 시퀀스 확장 (시간 차원)
        self.time_projection = layers.Dense(time_dim, activation='relu')
        
        # 멀티헤드 어텐션 블록
        self.transformer_blocks = []
        for _ in range(num_layers):
            self.transformer_blocks.append(
                TransformerBlock(audio_token_dim, num_heads, audio_token_dim * 4, dropout_rate)
            )
        
        # 오디오 차원 (주파수, 볼륨, 시간) 투영 레이어
        self.freq_projection = layers.Dense(frequency_dim, activation='tanh')
        self.volume_projection = layers.Dense(volume_dim, activation='sigmoid')
        self.time_sequence_layer = layers.Dense(time_dim, activation=None)
    
    def call(self, inputs, training=False):
        # 텍스트 임베딩 처리
        x = self.text_input_layer(inputs)
        x = self.text_dense(x)
        
        # 1D -> 2D 변환 (시퀀스)
        x = tf.expand_dims(x, axis=1)  # (batch_size, 1, audio_token_dim)
        
        # 시간 차원으로 확장 (batch_size, time_dim, audio_token_dim)
        time_weights = self.time_projection(x)
        time_weights = tf.nn.softmax(time_weights, axis=1)
        
        # 확장된 시퀀스 생성
        x = tf.tile(x, [1, self.time_dim, 1])  # 시간 차원으로 복제
        
        # 트랜스포머 처리
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        
        # 3차원 출력 (주파수, 볼륨, 시간)
        freq_output = self.freq_projection(x)  # (batch_size, time_dim, frequency_dim)
        volume_output = self.volume_projection(x)  # (batch_size, time_dim, volume_dim)
        
        # 최종 출력 형태 (batch_size, time_dim, frequency_dim + volume_dim)
        return tf.concat([freq_output, volume_output], axis=-1)
    
    def get_config(self):
        config = super(TextToAudioTokenizer, self).get_config()
        config.update({
            'text_embed_dim': self.text_embed_dim,
            'audio_token_dim': self.audio_token_dim,
            'frequency_dim': self.frequency_dim,
            'volume_dim': self.volume_dim,
            'time_dim': self.time_dim,
        })
        return config

class TransformerBlock(layers.Layer):
    """트랜스포머 블록 레이어"""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        # 셀프 어텐션
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, 
            training=training
        )
        attention_output = self.dropout1(attention_output, training=training)
        # 첫 번째 잔차 연결과 정규화
        out1 = self.layernorm1(inputs + attention_output)
        
        # 피드포워드 네트워크
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        # 두 번째 잔차 연결과 정규화
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
        })
        return config

def build_text_to_audio_model(diffusion_model_path, text_embed_dim=512, 
                             frequency_dim=1024, volume_dim=128, time_dim=2048,
                             frame_size=2048):
    """텍스트-오디오 변환 통합 모델 구축"""
    # 텍스트 입력
    text_input = layers.Input(shape=(text_embed_dim,))
    
    # 텍스트-오디오 토크나이저 (트랜스포머)
    tokenizer = TextToAudioTokenizer(
        text_embed_dim=text_embed_dim,
        frequency_dim=frequency_dim,
        volume_dim=volume_dim,
        time_dim=time_dim
    )
    
    # 토큰화된 오디오 특성
    audio_tokens = tokenizer(text_input)
    
    # 디퓨전 모델 로드
    try:
        diffusion_model = keras.models.load_model(diffusion_model_path)
        diffusion_model.trainable = False  # 디퓨전 모델은 고정
    except Exception as e:
        print(f"디퓨전 모델 로드 오류: {e}")
        # 디퓨전 모델이 없으면 임시 레이어로 대체
        diffusion_model = layers.Dense(frame_size, activation='tanh')
        
    # 최종 오디오 생성
    # 여기서는 디퓨전 모델을 직접 통합하지 않고, 
    # 트랜스포머로 생성된 토큰을 반환
    # 실제 오디오 생성은 generate_audio 함수에서 처리
    
    # 모델 생성
    model = keras.Model(inputs=text_input, outputs=audio_tokens)
    return model

def train_text_to_audio(text_data, audio_files, epochs=100, batch_size=16, model_path="text_to_audio_model.keras"):
    """텍스트-오디오 모델 학습"""
    # 텍스트 임베딩 변환
    text_embeddings = []
    for text in tqdm(text_data, desc="텍스트 임베딩 처리"):
        embedding = text_to_embedding(text)
        embedding = adjust_embedding_dimension(embedding, target_dim=512)
        text_embeddings.append(embedding[0])
    text_embeddings = np.array(text_embeddings)
    
    # 오디오 데이터 처리
    audio_features = []
    
    for audio_file in tqdm(audio_files, desc="오디오 특성 추출"):
        try:
            # 웨이블릿 변환
            features = Data(audio_file).transform()
            audio_features.append(features)
        except Exception as e:
            print(f"오디오 파일 처리 오류 {audio_file}: {e}")
    
    # 특성 데이터 통합 및 정규화
    x_train = text_embeddings
    y_train = np.array(audio_features)
    
    print(f"학습 데이터 형태: x={x_train.shape}, y={y_train.shape}")
    
    # 모델 구축
    model = build_text_to_audio_model(
        diffusion_model_path="diffusion_model",
        text_embed_dim=512,
        frequency_dim=1024,
        volume_dim=128,
        time_dim=2048
    )
    
    # 옵티마이저 및 손실 함수 설정
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='mse'
    )
    
    # 학습
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        model_path, save_best_only=True, monitor='val_loss'
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback, early_stopping]
    )
    
    # 모델 저장
    model.save(model_path)
    print(f"모델 저장 완료: {model_path}")
    
    return model, history

def generate_audio_from_text(text, tokenizer_model_path, diffusion_model_path, 
                           diffusion_params_path="diffusion_params.json",
                           output_path="generated.wav",
                           sample_rate=48000):
    """텍스트로부터 오디오 생성"""
    # 텍스트 임베딩
    text_embedding = text_to_embedding(text)
    text_embedding = adjust_embedding_dimension(text_embedding, target_dim=512)
    
    # 배치 차원 확인 및 조정 (모델 입력에 맞게)
    if len(text_embedding.shape) == 2:
        # 이미 배치 형태
        pass
    else:
        # 단일 임베딩을 배치로 변환
        text_embedding = np.expand_dims(text_embedding, axis=0)
    
    # 토크나이저 모델 로드
    try:
        tokenizer_model = keras.models.load_model(
            tokenizer_model_path,
            custom_objects={
                "TextToAudioTokenizer": TextToAudioTokenizer,
                "TransformerBlock": TransformerBlock
            }
        )
        
        # 텍스트에서 오디오 토큰 생성
        audio_tokens = tokenizer_model.predict(text_embedding)
        
    except Exception as e:
        print(f"토크나이저 모델 로드 또는 예측 오류: {e}")
        return None
    
    # 디퓨전 모델 및 유틸리티 함수 로드
    from main import DiffusionModel, sample_from_diffusion, inverse_transform, save_audio
    
    # 디퓨전 파라미터 로드 (예외 처리 추가)
    try:
        with open(diffusion_params_path, 'r') as f:
            params = json.load(f)
    except FileNotFoundError:
        print(f"디퓨전 파라미터 파일을 찾을 수 없습니다: {diffusion_params_path}")
        # 기본 파라미터 설정
        params = {'timesteps': 1000, 'beta_schedule': 'linear'}
    except json.JSONDecodeError:
        print(f"디퓨전 파라미터 파일 형식이 잘못되었습니다: {diffusion_params_path}")
        # 기본 파라미터 설정
        params = {'timesteps': 1000, 'beta_schedule': 'linear'}
    
    # 디퓨전 모델 초기화
    diffusion = DiffusionModel(
        timesteps=params.get('timesteps', 1000),
        beta_schedule=params.get('beta_schedule', 'linear')
    )
    
    # 디퓨전 파라미터 설정
    if 'betas' in params:
        diffusion.betas = np.array(params['betas'], dtype=np.float32)
    if 'alphas' in params:
        diffusion.alphas = np.array(params['alphas'], dtype=np.float32)
    if 'alphas_cumprod' in params:
        diffusion.alphas_cumprod = np.array(params['alphas_cumprod'], dtype=np.float32)
        diffusion.alphas_cumprod_prev = np.append(np.float32(1.0), diffusion.alphas_cumprod[:-1])
    
    # 추가 필요한 파라미터 계산
    if not hasattr(diffusion, 'sqrt_alphas_cumprod') or diffusion.sqrt_alphas_cumprod is None:
        diffusion.sqrt_alphas_cumprod = np.sqrt(diffusion.alphas_cumprod).astype(np.float32)
    if not hasattr(diffusion, 'sqrt_one_minus_alphas_cumprod') or diffusion.sqrt_one_minus_alphas_cumprod is None:
        diffusion.sqrt_one_minus_alphas_cumprod = np.sqrt(1. - diffusion.alphas_cumprod).astype(np.float32)
    
    # 디퓨전 모델 로드
    try:
        diffusion_model = keras.models.load_model(diffusion_model_path)
    except Exception as e:
        print(f"디퓨전 모델 로드 오류: {e}")
        return None
    
    # 오디오 토큰으로부터 웨이블릿 데이터 생성
    # 샘플링을 위한 shape 설정 (주의: audio_tokens의 형태를 사용)
    print(f"오디오 토큰 형태: {audio_tokens.shape}")
    
    # 샘플링
    try:
        # 합리적인 스텝 수와 노이즈 비율 적용
        steps = 100
        eta = 0.3
        
        generated = sample_from_diffusion(
            diffusion_model, diffusion, audio_tokens.shape, 
            steps=steps, eta=eta, num_samples=1
        )
        
        # 웨이블릿 역변환
        audio_samples = inverse_transform(generated.numpy())
        
        # 오디오 저장 
        if audio_samples is not None and len(audio_samples) > 0:
            save_audio(audio_samples, output_path, sample_rate=sample_rate)
            print(f"오디오 생성 완료: {output_path}")
            return audio_samples
        else:
            print("오디오 생성 실패: 유효한 샘플이 생성되지 않았습니다.")
            return None
            
    except Exception as e:
        print(f"오디오 생성 중 오류 발생: {e}")
        return None
