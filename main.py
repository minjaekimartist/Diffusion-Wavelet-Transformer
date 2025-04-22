import concurrent
from concurrent.futures import ThreadPoolExecutor
import glob
import platform
import json
import keras
import logging
import numpy as np
import os
import pywt
import struct
import sys
import tensorflow as tf
from tqdm import tqdm
import wave

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 혼합 정밀도 정책 설정 (Apple Silicon에 최적화)
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
logger.info(f"정밀도 정책: {policy.name}")

# 설정 값
sample_size = 2048
WAVELET_TYPE = 'db4'  # 웨이블릿 타입 (db1, db4, sym4 등)
MAX_FILES = 100       # 처리할 최대 파일 수
DEFAULT_SAMPLE_RATE = 48000

def check_available_devices():
    """사용 가능한 장치 목록 확인"""
    devices = tf.config.list_physical_devices()
    for device in devices:
        logger.info(f"사용 가능한 장치: {device.name}, 유형: {device.device_type}")
    return devices

def configure_npu():
    """NPU 설정"""
    try:
        # Apple Silicon의 경우 Metal 장치를 사용
        if platform.system() == 'Darwin' and platform.processor() == 'arm':
            logger.info("Apple Silicon 감지, Metal 백엔드 확인 중...")
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"Apple Neural Engine/GPU: {len(gpus)}개 발견")
                # 메모리 증가 허용
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                # Apple은 기본적으로 Neural Engine을 사용함
                return True
            else:
                logger.warning("Apple Neural Engine을 찾을 수 없습니다.")
                return False
        
        # 다른 NPU 장치 (비 Apple)
        npus = tf.config.list_physical_devices('NPU')
        if npus:
            logger.info(f"사용 가능한 NPU: {len(npus)}개 발견")
            # 메모리 증가 허용
            for npu in npus:
                tf.config.experimental.set_memory_growth(npu, True)
            # 첫 번째 NPU만 사용 (필요에 따라 변경)
            tf.config.set_visible_devices(npus[0], 'NPU')
            return True
        else:
            logger.warning("NPU 장치를 찾을 수 없습니다.")
            return False
    except Exception as e:
        logger.error(f"NPU 설정 중 오류 발생: {str(e)}")
        return False

@keras.saving.register_keras_serializable()
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Data:
    def __init__(self, path: str):
        try:
            wave_file = wave.open(path, 'r')
        except wave.Error:
            logger.error(f'Error: Invalid wave file: {path}')
            raise wave.Error
        except FileNotFoundError:
            logger.error(f'Error: File not found: {path}')
            raise FileNotFoundError
            
        self.sample_rate = wave_file.getframerate()
        data = wave_file.readframes(wave_file.getnframes())
        self.samples = []
        sample_width = wave_file.getsampwidth()
        channels = wave_file.getnchannels()
        
        # 샘플 너비에 따른 최대값 설정
        max_values = {1: 128, 2: 32768, 3: 8388608, 4: 2147483648}
        max_value = max_values.get(sample_width, 2147483648)
            
        # 더 효율적인 데이터 처리
        bytes_per_sample = sample_width * channels
        for i in range(0, len(data) // bytes_per_sample):
            sample = data[i * bytes_per_sample: (i + 1) * bytes_per_sample]
            int_value = int.from_bytes(sample, byteorder='little', signed=True)
            float_value = float(int_value) / max_value
            self.samples.append(float_value)
            
        wave_file.close()
        logger.info(f'Loaded wave file: {path} (길이: {len(self.samples)})')

    def transform(self) -> list[np.ndarray]:    
        """웨이블릿 변환 병렬 처리"""
        result = []
        segments = len(self.samples) // sample_size
        segments_list = []
        
        # 세그먼트 미리 준비
        for i in range(segments):
            segment = self.samples[i * sample_size: (i + 1) * sample_size]
            if len(segment) == sample_size:  # 세그먼트가 짧으면 건너뛰기
                segments_list.append(segment)
        
        def process_segment(segment):
            try:
                # 노이즈 추가로 증강 (약 10%)
                if np.random.random() < 0.1:
                    noise_level = np.random.uniform(0.001, 0.01)
                    segment = segment + np.random.normal(0, noise_level, segment.shape)
                
                # 진폭 변화 (약 20%)
                if np.random.random() < 0.2:
                    gain = np.random.uniform(0.8, 1.2)
                    segment = segment * gain
                    
                # 웨이블릿 변환 - 기존 코드와 동일
                cA, cD = pywt.dwt(segment, WAVELET_TYPE)
                
                # 정규화 추가
                cA = (cA - np.mean(cA)) / (np.std(cA) + 1e-8)
                cD = (cD - np.mean(cD)) / (np.std(cD) + 1e-8)
                
                if np.any(np.isnan(cA)) or np.any(np.isnan(cD)):
                    return None
                return np.array([cA, cD])
            except Exception as e:
                logger.error(f"세그먼트 처리 중 오류 발생: {str(e)}")
                return None
        
        # CPU 코어 수에 따라 워커 수 조정
        num_workers = min(32, os.cpu_count() or 4)
        
        # 병렬 처리
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # map 대신 submit/as_completed 사용하여 결과 즉시 처리
            futures = [executor.submit(process_segment, segment) for segment in segments_list]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(segments_list), desc="웨이블릿 변환"):
                res = future.result()
                if res is not None:
                    result.append(res)
        
        return result

def load_data_parallel(folder: str, max_files=MAX_FILES):
    """여러 WAV 파일을 병렬로 로드하고 처리"""
    wav_files = glob.glob(os.path.join(folder, '*.wav'))
    wav_files = wav_files[:max_files]  # 파일 수 제한
    
    logger.info(f"총 {len(wav_files)} 개의 WAV 파일을 처리합니다.")
    data = []
    
    def process_file(wav_file):
        try:
            audio_data = Data(wav_file)
            return audio_data.transform()
        except Exception as e:
            logger.error(f"파일 처리 중 오류 발생: {wav_file}, {str(e)}")
            return []
    
    # 병렬 처리
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(process_file, wav_files), total=len(wav_files)))
    
    # 결과 합치기
    for result in results:
        data.extend(result)
    
    logger.info(f"총 {len(data)} 개의 세그먼트를 로드했습니다.")
    return np.array(data)

class DiffusionModel:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, beta_schedule='cosine', s=0.008):
        self.timesteps = timesteps

        # 베타 스케줄 선택
        if beta_schedule == 'linear':
            self.betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)
        elif beta_schedule == 'quadratic':
            self.betas = np.square(np.linspace(np.sqrt(beta_start), np.sqrt(beta_end), timesteps, dtype=np.float32))
        elif beta_schedule == 'cosine':
            # cosine 스케줄 (Improved DDPM 논문 참조)
            steps = timesteps + 1
            x = np.linspace(0, timesteps, steps, dtype=np.float32)
            alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return np.clip(betas, 0, 0.999)
        else:
            self.betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)

        # 나머지 파라미터 계산
        self.alphas = (1. - self.betas).astype(np.float32)
        self.alphas_cumprod = np.cumprod(self.alphas).astype(np.float32)
        self.alphas_cumprod_prev = np.append(np.float32(1.0), self.alphas_cumprod[:-1])

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod).astype(np.float32)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1. - self.alphas_cumprod).astype(np.float32)
        self.sqrt_recip_alphas = np.sqrt(1. / self.alphas).astype(np.float32)
        
        # 사후 분산과 평균 계산
        self.posterior_variance = (self.betas * (1. - self.alphas_cumprod_prev) / 
                                  (1. - self.alphas_cumprod)).astype(np.float32)
        self.posterior_mean_coef1 = (self.betas * np.sqrt(self.alphas_cumprod_prev) / 
                                    (1. - self.alphas_cumprod)).astype(np.float32)
        self.posterior_mean_coef2 = ((1. - self.alphas_cumprod_prev) * np.sqrt(self.alphas) / 
                                    (1. - self.alphas_cumprod)).astype(np.float32)
    
    def q_sample(self, x_0, t, noise=None):
        """forward 과정 (노이즈 추가)"""
        # 입력 텐서 타입 확인 및 통일
        dtype = x_0.dtype
        
        if noise is None:
            noise = tf.random.normal(shape=tf.shape(x_0), dtype=dtype)
        else:
            # 노이즈도 동일한 데이터 타입으로 변환
            noise = tf.cast(noise, dtype)

        # 알파 값들도 동일한 데이터 타입으로 변환
        sqrt_alphas_cumprod_t = tf.cast(tf.gather(self.sqrt_alphas_cumprod, t), dtype)
        sqrt_one_minus_alphas_cumprod_t = tf.cast(
            tf.gather(self.sqrt_one_minus_alphas_cumprod, t), dtype)
        
        # 차원 확장
        while len(sqrt_alphas_cumprod_t.shape) < len(x_0.shape):
            sqrt_alphas_cumprod_t = tf.expand_dims(sqrt_alphas_cumprod_t, -1)
            sqrt_one_minus_alphas_cumprod_t = tf.expand_dims(sqrt_one_minus_alphas_cumprod_t, -1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, denoise_model, x_0, t, noise=None, loss_type="huber"):
        """모델 학습을 위한 손실 함수"""
        if noise is None:
            noise = tf.random.normal(shape=tf.shape(x_0))
        
        x_noisy = self.q_sample(x_0, t, noise)
        predicted_noise = denoise_model([x_noisy, t], training=True)
        
        if loss_type == "l1":
            loss = tf.reduce_mean(tf.abs(noise - predicted_noise))
        elif loss_type == "huber":
            loss = tf.keras.losses.Huber(delta=0.1)(noise, predicted_noise)
        else:  # l2
            loss = tf.reduce_mean(tf.square(noise - predicted_noise))
            
        return loss
    
    def p_sample(self, model, x, t, t_index, eta=0.0):
        """단일 샘플링 스텝 (노이즈 제거)"""
        if platform.system() == 'Darwin' and platform.processor() == 'arm':
            device = "/device:GPU:0"
        elif tf.config.list_physical_devices('NPU'):
            device = "/device:NPU:0"
        elif tf.config.list_physical_devices('GPU'):
            device = "/device:GPU:0"
        else:
            device = "/device:CPU:0"
        
        with tf.device(device):
            # 타입 변환
            x = tf.cast(x, tf.float16)
            
            # 예측된 노이즈 계산
            pred_noise = model([x, t], training=False)
            
            # 알파 값들 준비 (float16 사용)
            alpha = tf.constant(self.alphas[t_index], dtype=tf.float16)
            alpha_cumprod = tf.constant(self.alphas_cumprod[t_index], dtype=tf.float16)
            alpha_cumprod_prev = tf.constant(self.alphas_cumprod_prev[t_index], dtype=tf.float16)
            beta = tf.constant(self.betas[t_index], dtype=tf.float16)
            
            # 상수 준비
            one = tf.constant(1.0, dtype=tf.float16)
            
            # 현재 예측을 기반으로 평균 계산
            pred_x0 = (x - tf.sqrt(one - alpha_cumprod) * pred_noise) / tf.sqrt(alpha_cumprod)
            pred_x0 = tf.clip_by_value(pred_x0, -1.0, 1.0)  # 안정성을 위한 클리핑
            
            # 평균 계산
            c1 = tf.sqrt(one - beta) * (x - beta * pred_x0 / tf.sqrt(one - alpha_cumprod)) / tf.sqrt(one - alpha_cumprod)
            c2 = tf.sqrt(beta) * pred_x0
            mean = c1 + c2
            
            # 분산 계산 및 노이즈 샘플링
            var = tf.clip_by_value(
                beta * (one - alpha_cumprod_prev) / (one - alpha_cumprod), 
                0.0, 1.0  # 안전한 범위로 클리핑
            )
            noise = tf.random.normal(shape=tf.shape(x), dtype=tf.float16)  # float16 사용
            
            # 최종 샘플 계산 (오버플로우 방지)
            x_prev = mean + tf.clip_by_value(tf.sqrt(var) * eta * noise, -1.0, 1.0)
            
            return x_prev

def build_unet_model(input_shape, time_embedding_dim=64, base_filters=32, depth=3, attention_res=[2]):
    """
    개선된 U-Net 모델 구축 함수
    
    Args:
        input_shape: 입력 데이터 형태
        time_embedding_dim: 시간 임베딩 차원
        base_filters: 기본 필터 수
        depth: U-Net 깊이
        attention_res: 어텐션 레이어를 적용할 해상도 레벨
    """
    # 시간 임베딩
    time_input = keras.layers.Input(shape=(1,))
    time_embedding = keras.layers.Embedding(1000, time_embedding_dim)(time_input)
    time_embedding = keras.layers.Dense(time_embedding_dim)(time_embedding)
    time_embedding = keras.layers.Activation('swish')(time_embedding)
    time_embedding = keras.layers.Dense(time_embedding_dim)(time_embedding)
    
    # 데이터 입력
    inputs = keras.layers.Input(shape=input_shape)
    
    # 스케일된 시간 임베딩 준비
    t_emb = keras.layers.Dense(input_shape[0] * input_shape[1])(time_embedding)
    t_emb = keras.layers.Reshape((input_shape[0], input_shape[1], 1))(t_emb)
    
    # 입력과 시간 임베딩 결합
    h = keras.layers.Concatenate(axis=-1)([inputs, t_emb])
    
    # 다운샘플링 경로
    skip_connections = []
    
    h = keras.layers.Conv2D(base_filters, 3, padding='same')(h)
    skip_connections.append(h)
    
    # 인코더 부분 (다운샘플링)
    for i in range(depth):
        filters = base_filters * (2 ** i)
        
        # 레지듀얼 블록
        h_res = keras.layers.Conv2D(filters, 3, padding='same')(h)
        h_res = keras.layers.BatchNormalization()(h_res)
        h_res = keras.layers.Activation('swish')(h_res)
        h_res = keras.layers.Dropout(0.1)(h_res)  # 10% 드롭아웃
        h_res = keras.layers.Conv2D(filters, 3, padding='same')(h_res)
        h_res = keras.layers.BatchNormalization()(h_res)
        
        # 입력 채널 수와 출력 채널 수가 다를 수 있으므로 1x1 컨볼루션으로 맞춤
        if h.shape[-1] != filters:
            h = keras.layers.Conv2D(filters, 1, padding='same')(h)
            
        h_res = keras.layers.Add()([h, h_res])  # 레지듀얼 연결
        h = h_res
        
        # 셀프 어텐션 레이어 (선택적)
        if i in attention_res:
            h = attention_block(h, filters)
        
        skip_connections.append(h)
        
        # 다운샘플링
        if i < depth - 1:  # 마지막 블록에서는 다운샘플링 없음
            h = keras.layers.Conv2D(filters * 2, 3, strides=2, padding='same')(h)
            h = keras.layers.BatchNormalization()(h)
            h = keras.layers.Activation('swish')(h)
    
    # 디코더 부분 (업샘플링)
    for i in reversed(range(depth)):
        filters = base_filters * (2 ** i)
        
        # 업샘플링
        if i < depth - 1:  # 첫 번째 블록에서는 업샘플링 없음
            h = keras.layers.Conv2DTranspose(filters, 3, strides=2, padding='same')(h)
            h = keras.layers.BatchNormalization()(h)
            h = keras.layers.Activation('swish')(h)
        
        # 스킵 연결 전에 크기 조정
        skip = skip_connections[i]
        
        # 텐서 크기 확인 및 조정
        if h.shape[1] != skip.shape[1] or h.shape[2] != skip.shape[2]:
            # 크기 불일치 시, 리사이즈 적용
            h = keras.layers.Resizing(
                skip.shape[1], 
                skip.shape[2],
                interpolation="bilinear"
            )(h)
            logger.info(f"텐서 크기 조정: {h.shape} -> {skip.shape}")
        
        # 스킵 연결
        h = keras.layers.Concatenate(axis=-1)([h, skip])
        
        # 레지듀얼 블록
        h_res = keras.layers.Conv2D(filters, 3, padding='same')(h)
        h_res = keras.layers.BatchNormalization()(h_res)
        h_res = keras.layers.Activation('swish')(h_res)
        h_res = keras.layers.Conv2D(filters, 3, padding='same')(h_res)
        h_res = keras.layers.BatchNormalization()(h_res)
        
        # 입력과 출력 채널 수가 다를 수 있으므로 1x1 컨볼루션으로 맞춤
        h_skip = keras.layers.Conv2D(filters, 1, padding='same')(h)
        h_res = keras.layers.Add()([h_skip, h_res])  # 레지듀얼 연결
        h = h_res
        
        # 셀프 어텐션 레이어 (선택적)
        if i in attention_res:
            h = attention_block(h, filters)
    
    # 출력 레이어
    outputs = keras.layers.Conv2D(input_shape[-1], 1, activation=None)(h)
    
    # 모델 생성
    model = keras.models.Model([inputs, time_input], outputs)
    return model

def attention_block(x, filters):
    """셀프 어텐션 블록"""
    h = x
    batch_size, height, width, channels = x.shape
    
    # 쿼리, 키, 값 생성
    q = keras.layers.Conv2D(filters // 8, 1, padding='same')(h)
    k = keras.layers.Conv2D(filters // 8, 1, padding='same')(h)
    v = keras.layers.Conv2D(filters, 1, padding='same')(h)
    
    # 리셰이프
    q = keras.layers.Reshape((-1, filters // 8))(q)
    k = keras.layers.Reshape((-1, filters // 8))(k)
    v = keras.layers.Reshape((-1, filters))(v)
    
    # 어텐션 맵 계산
    q = keras.layers.Permute((2, 1))(q)  # 전치
    attn = keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1]))([k, q])
    attn = keras.layers.Lambda(lambda x: x / np.sqrt(filters // 8))(attn)
    attn = keras.layers.Softmax(axis=-1)(attn)
    
    # 값과 결합
    output = keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1]))([attn, v])
    output = keras.layers.Reshape((height, width, filters))(output)
    
    # 원본과 결합
    output = keras.layers.Conv2D(channels, 1, padding='same')(output)
    return keras.layers.Add()([x, output * 0.1])  # 스케일링 적용

def train(folder: str, timesteps=1000, epochs=100, batch_size=16, beta_schedule='cosine'):
    """데이터 학습 함수"""
    # 데이터 로드
    data = load_data_parallel(folder, max_files=MAX_FILES)
    
    if len(data) == 0:
        logger.error("처리할 데이터가 없습니다. 데이터 로드에 실패했습니다.")
        return
    
    logger.info(f"데이터 형태: {data.shape}")
    
    # 모델 설정
    diffusion = DiffusionModel(timesteps=timesteps, beta_schedule=beta_schedule)
    input_shape = (data.shape[1], data.shape[2], 1)  # 채널 차원 추가
    
    # 데이터를 올바른 형태로 변환 (채널 추가)
    data = np.expand_dims(data, axis=-1)
    # 훈련 데이터를 메모리에 올림
    samples = tf.cast(tf.convert_to_tensor(data), tf.float32)
    # 체크포인트 디렉토리 생성
    os.makedirs('checkpoints', exist_ok=True)
    
    # 학습 시작
    strategy = None
    gpus = tf.config.list_physical_devices('GPU')
    
    if len(gpus) > 1:
        # 다중 GPU 사용
        logger.info(f"{len(gpus)}개의 GPU를 사용한 분산 학습을 설정합니다.")
        strategy = tf.distribute.MirroredStrategy()
        batch_size = batch_size * len(gpus)  # 배치 크기 조정
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/device:GPU:0" if gpus else "/device:CPU:0")
    
    with strategy.scope():
        # 모델 구축
        model = build_unet_model(
            input_shape=input_shape, 
            time_embedding_dim=128, 
            base_filters=64, 
            depth=4, 
            attention_res=[1, 2, 3]
        )
        
        # 옵티마이저 및 학습률 스케줄러 설정
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-5,
            decay_steps=epochs * (len(data) // batch_size),
            alpha=0.01
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=optimizer, loss='mse')
    
        def train_step_fn(batch_indices):
            batch_data = tf.gather(samples, batch_indices)
            batch_data = tf.cast(batch_data, tf.float16)
            
            accumulated_gradients = None
            num_accumulations = 4  # 4번 누적 (가상 배치 크기 = 4 * batch_size)
            
            # 그래디언트 누적
            for i in range(num_accumulations):
                # 각 배치 분할
                start_idx = i * (batch_size // num_accumulations)
                end_idx = (i + 1) * (batch_size // num_accumulations)
                sub_batch = batch_data[start_idx:end_idx]
                
                t = tf.random.uniform(
                    shape=[sub_batch.shape[0]], 
                    minval=0, 
                    maxval=timesteps, 
                    dtype=tf.int32
                )
                
                noise = tf.random.normal(shape=sub_batch.shape, dtype=tf.float16)
                x_noisy = diffusion.q_sample(sub_batch, t, noise)
                t_expanded = tf.expand_dims(t, -1)
                
                with tf.GradientTape() as tape:
                    predicted_noise = model([x_noisy, t_expanded], training=True)
                    loss = tf.reduce_mean(tf.square(noise - predicted_noise)) / num_accumulations
                
                gradients = tape.gradient(loss, model.trainable_variables)
                
                # 그래디언트 누적
                if accumulated_gradients is None:
                    accumulated_gradients = gradients
                else:
                    accumulated_gradients = [accu_grad + grad for accu_grad, grad in zip(accumulated_gradients, gradients)]
            
            # 클리핑 및 적용
            accumulated_gradients, _ = tf.clip_by_global_norm(accumulated_gradients, clip_norm=1.0)
            optimizer.apply_gradients(zip(accumulated_gradients, model.trainable_variables))
            return loss * num_accumulations  # 원래 손실 크기로 반환
        
        if platform.system() == 'Darwin' and platform.processor() == 'arm':
            # Apple Silicon에서는 tf.function 데코레이터에서 jit_compile 비활성화
            @tf.function(jit_compile=False, reduce_retracing=True)
            def train_step(batch_indices):
                per_replica_losses = strategy.run(train_step_fn, args=(batch_indices,))
                return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        else:
            # 다른 플랫폼에서는 jit_compile 활성화
            @tf.function(jit_compile=True, reduce_retracing=True)
            def train_step(batch_indices):
                per_replica_losses = strategy.run(train_step_fn, args=(batch_indices,))
                return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    
    dataset = tf.data.Dataset.range(len(samples))
    dataset = dataset.shuffle(buffer_size=min(len(samples), 10000))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    warmup_epochs = 5
    initial_lr = 1e-5
    # 학습 시작
    for epoch in range(epochs):
        logger.info(f"에포크 {epoch+1}/{epochs}")
        # 학습률 조정 (워밍업)
        if epoch < warmup_epochs:
        # 첫 5에포크 동안 학습률을 서서히 증가
            lr = initial_lr * (epoch + 1) / warmup_epochs
            keras.backend.set_value(optimizer.learning_rate, lr)

        # 사용자 정의 훈련 루프 실행
        epoch_loss = 0
        num_batches = 0
        
        for batch_indices in tqdm(dataset, desc=f"Epoch {epoch+1}/{epochs}"):
            loss = train_step(batch_indices)
            epoch_loss += loss
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        logger.info(f"에포크 {epoch+1} 평균 손실: {avg_loss:.6f}")
        
        # 모델 저장 (주기적으로)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.save(f'checkpoints/diffusion_model_epoch_{epoch+1}.keras')
            logger.info(f"모델 저장됨: checkpoints/diffusion_model_epoch_{epoch+1}.keras")
    
    # 최종 모델 저장
    model.save('diffusion_model.keras')
    logger.info("최종 모델 저장됨: diffusion_model.keras")
    
    # 디퓨전 파라미터 저장
    diffusion_params = {
        'timesteps': diffusion.timesteps,
        'betas': diffusion.betas.tolist(),
        'alphas': diffusion.alphas.tolist(),
        'alphas_cumprod': diffusion.alphas_cumprod.tolist(),
        'beta_schedule': beta_schedule
    }
    
    with open('diffusion_params.json', 'w') as f:
        json.dump(diffusion_params, f)
    logger.info("디퓨전 파라미터 저장됨: diffusion_params.json")

def sample_from_diffusion(model, diffusion, shape, steps=1000, eta=0, num_samples=4):
    """여러 샘플을 병렬로 생성"""
    # 배치 크기를 num_samples로 설정
    batch_shape = (num_samples,) + shape[1:]
    x = tf.random.normal(batch_shape, dtype=tf.float32)
    
    # 반복적인 노이즈 제거
    for i in tqdm(reversed(range(steps)), desc="샘플링"):
        t = tf.ones((batch_shape[0],), dtype=tf.int32) * i
        
        # 병렬 처리를 위한 배치 연산
        predicted_noise = model([x, t[:, None]], training=False)
        
        alpha = tf.constant(diffusion.alphas[i], dtype=tf.float32)
        alpha_cumprod = tf.constant(diffusion.alphas_cumprod[i], dtype=tf.float32)
        alpha_cumprod_prev = tf.constant(diffusion.alphas_cumprod_prev[i], dtype=tf.float32)
        
        one = tf.constant(1.0, dtype=tf.float32)
        sigma = eta * tf.sqrt((one - alpha_cumprod_prev) / (one - alpha_cumprod) * 
                             (one - alpha / alpha_cumprod_prev))
        
        c1 = tf.math.rsqrt(alpha)
        c2 = (one - alpha) / tf.sqrt(one - alpha_cumprod)
        
        x = c1 * (x - c2 * predicted_noise)
        
        if i > 0:
            noise = tf.random.normal(batch_shape, dtype=tf.float32)
            x = x + sigma * noise
    
    return x

def load_model(model_path='autoencoder.keras'):
    custom_objects = {
        'Sampling': Sampling
    }
    return keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

def inverse_transform(wavelet_data):
    """병렬 웨이블릿 역변환"""
    if wavelet_data.ndim == 4:
        if wavelet_data.shape[3] == 1:
            wavelet_data = wavelet_data[0, :, :, 0]
        else:
            wavelet_data = wavelet_data[0]
    elif wavelet_data.ndim == 3 and wavelet_data.shape[0] == 1:
        wavelet_data = wavelet_data[0]
        
    def process_segment(i):
        try:
            cA = wavelet_data[i, 0]
            cD = wavelet_data[i, 1]
            reconstructed = pywt.idwt(cA, cD, WAVELET_TYPE)
            if reconstructed is not None and not np.all(np.isnan(reconstructed)) and not np.all(reconstructed == 0):
                return reconstructed
        except Exception as e:
            logger.error(f"역변환 중 오류 발생 (세그먼트 {i}): {str(e)}")
        return None
    
    # 병렬 처리
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 4)) as executor:
        results = list(tqdm(
            executor.map(process_segment, range(wavelet_data.shape[0])), 
            total=wavelet_data.shape[0],
            desc="웨이블릿 역변환"
        ))
    
    # None이 아닌 결과들만 연결
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        logger.warning("유효한 오디오 세그먼트가 없습니다!")
        return np.zeros(1000)
    
    # 결과 합치기
    samples = np.concatenate(valid_results)
    return samples

def save_audio(samples, output_path, sample_rate=44100):
    samples = normalize_audio(samples)
    samples = np.clip(samples, -1.0, 1.0)
    
    with wave.open(output_path, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        for sample in samples:
            value = int(sample * 32767.0)
            data = struct.pack('<h', value)
            wav_file.writeframes(data)

def generate_audio(output_path='generated.wav', steps=100, eta=0.3, seed=None):
    """오디오 생성 함수"""
    logger.info(f"오디오 생성 시작: {output_path}")
    
    # 랜덤 시드 설정 (재현성)
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)
    
    # 모델 로드
    try:
        model = keras.models.load_model('diffusion_model.keras')
        logger.info("모델 로드 성공")
    except Exception as e:
        logger.error(f"모델 로드 실패: {str(e)}")
        return None
    
    # 디퓨전 파라미터 로드
    try:
        with open('diffusion_params.json', 'r') as f:
            params = json.load(f)
        
        beta_schedule = params.get('beta_schedule', 'linear')
        logger.info(f"파라미터 로드 성공 (beta_schedule: {beta_schedule})")
    except Exception as e:
        logger.error(f"파라미터 로드 실패: {str(e)}")
        params = {'timesteps': 1000}
        beta_schedule = 'linear'
    
    # 디퓨전 모델 초기화
    diffusion = DiffusionModel(timesteps=params['timesteps'], beta_schedule=beta_schedule)
    diffusion.betas = np.array(params['betas'], dtype=np.float32)
    diffusion.alphas = np.array(params['alphas'], dtype=np.float32)
    diffusion.alphas_cumprod = np.array(params['alphas_cumprod'], dtype=np.float32)
    diffusion.alphas_cumprod_prev = np.append(np.float32(1.0), diffusion.alphas_cumprod[:-1])
    diffusion.sqrt_alphas_cumprod = np.sqrt(diffusion.alphas_cumprod).astype(np.float32)
    diffusion.sqrt_one_minus_alphas_cumprod = np.sqrt(1. - diffusion.alphas_cumprod).astype(np.float32)

    # 생성할 데이터 형태
    shape = (1, 2048, 2, 1)  # 베치 크기, 높이, 너비, 채널

    # 샘플링
    logger.info(f"샘플링 시작 (스텝: {steps}, eta: {eta})")
    generated = sample_from_diffusion(model, diffusion, shape, steps=steps, eta=eta)
    logger.info("샘플링 완료")
    
    # 웨이블릿 역변환
    logger.info("웨이블릿 역변환 시작")
    audio_samples = inverse_transform(generated.numpy())
    logger.info(f"역변환 완료 (샘플 수: {len(audio_samples)})")
    
    # 오디오 저장
    save_audio(audio_samples, output_path, sample_rate=DEFAULT_SAMPLE_RATE)
    logger.info(f"오디오 저장 완료: {output_path}")
    
    # 결과 반환
    return {
        'samples': audio_samples,
        'path': output_path,
        'sample_rate': DEFAULT_SAMPLE_RATE
    }

def normalize_audio(samples, target_peak=0.9):
    max_amp = np.max(np.abs(samples))
    if max_amp > 0:
        gain = target_peak / max_amp
        samples = samples * gain
    return samples

if __name__ == '__main__':
    if platform.system() == 'Darwin' and platform.processor() == 'arm':
        # Apple Silicon 최적화
        logger.info("Apple Silicon 감지, Metal 최적화 적용 중...")
        try:
            # 혼합 정밀도 활성화
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            # Metal 성능 향상을 위한 설정
            os.environ['TF_METAL_ENABLED_GPU_COUNT'] = '1'  # GPU 수 설정
            os.environ['TF_METAL_DEVICE_FORCE_CPU'] = '0'  # Metal 사용
            os.environ['TF_METAL_DEBUG_ERROR_INSERTION'] = '0'  # 디버그 비활성화
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # 메모리 동적 할당
            
            # 스레드 수 최적화
            num_threads = min(32, os.cpu_count() * 2)  # 하이퍼스레딩 활용
            tf.config.threading.set_inter_op_parallelism_threads(num_threads)
            tf.config.threading.set_intra_op_parallelism_threads(num_threads)
            
            # 스레드 수 최적화
            num_threads = min(32, os.cpu_count() * 2)  # 하이퍼스레딩 활용
            tf.config.threading.set_inter_op_parallelism_threads(num_threads)
            tf.config.threading.set_intra_op_parallelism_threads(num_threads)
            
            # Metal에서 문제가 될 수 있는 최적화 옵션 비활성화
            tf.config.optimizer.set_experimental_options({
                "layout_optimizer": False,  # 레이아웃 최적화 비활성화
                "constant_folding": True,
                "shape_optimization": True,
                "remapping": False,  # 리매핑 비활성화
                "arithmetic_optimization": True,
                "dependency_optimization": True,
                "loop_optimization": True,
                "function_optimization": True,
                "debug_stripper": True,
                "auto_mixed_precision": True
            })
            
            # 메모리 최적화
            physical_devices = tf.config.list_physical_devices('GPU')
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
                
            logger.info(f"Apple Neural Engine 최적화 완료 (스레드: {num_threads})")
        except Exception as e:
            logger.warning(f"Apple Neural Engine 최적화 실패: {str(e)}")
    else:
        # 비 Apple 장치에서는 XLA 사용 가능
        tf.config.optimizer.set_jit(True)
        
        # 그래프 최적화 설정
        tf.config.optimizer.set_experimental_options({
            "layout_optimizer": True,
            "constant_folding": True,
            "shape_optimization": True,
            "remapping": True,
            "arithmetic_optimization": True,
            "dependency_optimization": True,
            "loop_optimization": True,
            "function_optimization": True,
            "debug_stripper": True,
            "auto_mixed_precision": True
        })
            
    check_available_devices()
    npu_available = configure_npu()
    if not npu_available:
        logger.info("GPU 사용 시도")
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"사용 가능한 GPU: {len(gpus)}개 발견")
                # 메모리 증가 허용
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                logger.warning("GPU 장치를 찾을 수 없습니다. CPU를 사용합니다.")
        except Exception as e:
            logger.error(f"GPU 설정 중 오류 발생: {str(e)}")
    if len(sys.argv) < 2:
        print("사용법:")
        print("  python main.py train [source_path] [timesteps] [epochs] [batch_size] [beta_schedule]")
        print("  python main.py generate [output_path] [steps] [eta] [seed]")
        sys.exit(1)
    elif sys.argv[1] == 'train':
        source_path = sys.argv[2] if len(sys.argv) > 2 else 'data/wav'
        steps = int(sys.argv[3]) if len(sys.argv) > 3 else 200
        epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 200
        batch_size = int(sys.argv[5]) if len(sys.argv) > 5 else 256
        beta_schedule = sys.argv[6] if len(sys.argv) > 6 else 'cosine'
        
        logger.info(f"학습 시작: {source_path}, timesteps={steps}, epochs={epochs}, batch_size={batch_size}, beta_schedule={beta_schedule}")
        train(source_path, timesteps=steps, epochs=epochs, batch_size=batch_size, beta_schedule=beta_schedule)
    elif sys.argv[1] == 'generate':
        output_path = sys.argv[2] if len(sys.argv) > 2 else 'generated.wav'
        steps = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        eta = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
        seed = int(sys.argv[5]) if len(sys.argv) > 5 else None
        
        logger.info(f"생성 시작: {output_path}, steps={steps}, eta={eta}, seed={seed}")
        result = generate_audio(output_path=output_path, steps=steps, eta=eta, seed=seed)
        
        if result:
            logger.info(f"생성 완료: {output_path} (샘플 수: {len(result['samples'])})")
        else:
            logger.error("생성 실패")
    else:
        print(f"알 수 없는 명령: {sys.argv[1]}")
        print("사용법:")
        print("  python main.py train [source_path] [timesteps] [epochs] [batch_size] [beta_schedule]")
        print("  python main.py generate [output_path] [steps] [eta] [seed]")
        sys.exit(1)