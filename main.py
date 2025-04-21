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
        result = []
        segments = len(self.samples) // sample_size
    
        for i in range(segments):
            segment = self.samples[i * sample_size: (i + 1) * sample_size]
            # 세그먼트가 짧으면 건너뛰기
            if len(segment) < sample_size:
                continue
                
            # 웨이블릿 변환
            cA, cD = pywt.dwt(segment, WAVELET_TYPE)
            # 유효한 결과인지 확인
            if np.any(np.isnan(cA)) or np.any(np.isnan(cD)):
                continue
                
            two_axis_vector = [cA, cD]
            result.append(np.array(two_axis_vector))
    
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
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, beta_schedule='linear'):
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
            alphas_cumprod = np.cos(((x / timesteps) + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = np.clip(betas, 0, 0.999)
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
        if noise is None:
            noise = tf.random.normal(shape=tf.shape(x_0), dtype=tf.float32)

        sqrt_alphas_cumprod_t = tf.cast(tf.gather(self.sqrt_alphas_cumprod, t), tf.float32)
        sqrt_one_minus_alphas_cumprod_t = tf.cast(
            tf.gather(self.sqrt_one_minus_alphas_cumprod, t), tf.float32)
        
        # 차원 확장
        while len(sqrt_alphas_cumprod_t.shape) < len(x_0.shape):
            sqrt_alphas_cumprod_t = tf.expand_dims(sqrt_alphas_cumprod_t, -1)
            sqrt_one_minus_alphas_cumprod_t = tf.expand_dims(sqrt_one_minus_alphas_cumprod_t, -1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, denoise_model, x_0, t, noise=None, loss_type="l2"):
        """모델 학습을 위한 손실 함수"""
        if noise is None:
            noise = tf.random.normal(shape=tf.shape(x_0))
        
        x_noisy = self.q_sample(x_0, t, noise)
        predicted_noise = denoise_model([x_noisy, t], training=True)
        
        if loss_type == "l1":
            loss = tf.reduce_mean(tf.abs(noise - predicted_noise))
        elif loss_type == "huber":
            loss = tf.keras.losses.Huber(delta=1.0)(noise, predicted_noise)
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
        
        with tf.device(device):  # GPU 사용 명시
            # 예측된 노이즈 계산
            pred_noise = model([x, t], training=False)
            
            # 알파 값들 준비
            alpha = tf.constant(self.alphas[t_index], dtype=tf.float32)
            alpha_cumprod = tf.constant(self.alphas_cumprod[t_index], dtype=tf.float32)
            alpha_cumprod_prev = tf.constant(self.alphas_cumprod_prev[t_index], dtype=tf.float32)
            beta = tf.constant(self.betas[t_index], dtype=tf.float32)
            
            # 상수 준비
            one = tf.constant(1.0, dtype=tf.float32)
            
            # 현재 예측을 기반으로 평균 계산
            pred_x0 = (x - tf.sqrt(one - alpha_cumprod) * pred_noise) / tf.sqrt(alpha_cumprod)
            pred_x0 = tf.clip_by_value(pred_x0, -1.0, 1.0)  # 안정성을 위한 클리핑
            
            # 평균 계산
            c1 = tf.sqrt(one - beta) * (x - beta * pred_x0 / tf.sqrt(one - alpha_cumprod)) / tf.sqrt(one - alpha_cumprod)
            c2 = tf.sqrt(beta) * pred_x0
            mean = c1 + c2
            
            # 분산 계산 및 노이즈 샘플링
            var = beta * (one - alpha_cumprod_prev) / (one - alpha_cumprod)
            noise = tf.random.normal(shape=tf.shape(x), dtype=tf.float32)
            
            # 최종 샘플 계산
            x_prev = mean + tf.sqrt(var) * eta * noise
            
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
    
    # 모델 구축
    model = build_unet_model(
        input_shape=input_shape, 
        time_embedding_dim=64, 
        base_filters=32, 
        depth=3, 
        attention_res=[1, 2]
    )
    
    # 옵티마이저 및 학습률 스케줄러 설정
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4,
        decay_steps=epochs * (len(data) // batch_size),
        alpha=0.1
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mse')
    
    # 모델 요약 출력
    model.summary()
    
    # 훈련 데이터를 메모리에 올림
    samples = data
    
    # 체크포인트 디렉토리 생성
    os.makedirs('checkpoints', exist_ok=True)
    
    # 학습 시작
    for epoch in range(epochs):
        logger.info(f"에포크 {epoch+1}/{epochs}")
        total_loss = 0
        num_batches = len(samples) // batch_size
        
        # 배치 학습
        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}"):
            batch = samples[batch_idx*batch_size:(batch_idx+1)*batch_size]
            batch = tf.convert_to_tensor(batch, dtype=tf.float32)

            # 랜덤 타임스텝 선택
            t = tf.random.uniform(
                shape=[batch_size], 
                minval=0, 
                maxval=timesteps, 
                dtype=tf.int32
            )
            
            # 그래디언트 계산 및 적용
            with tf.GradientTape() as tape:
                noise = tf.random.normal(shape=batch.shape)
                x_noisy = diffusion.q_sample(batch, t, noise)

                t_expanded = tf.expand_dims(t, -1)
                predicted_noise = model([x_noisy, t_expanded], training=True)
                
                # 예측 형태 확인 및 조정
                if noise.shape != predicted_noise.shape:
                    predicted_noise = tf.slice(
                        predicted_noise, 
                        [0, 0, 0, 0], 
                        [batch_size, noise.shape[1], noise.shape[2], noise.shape[3]]
                    )
                
                # 손실 계산 (L2 손실)
                loss = tf.reduce_mean(tf.square(noise - predicted_noise))
            
            # 그래디언트 적용
            gradients = tape.gradient(loss, model.trainable_variables)
            
            # 그래디언트 클리핑 (NaN 방지)
            gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in gradients]
            
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss += loss.numpy()
        
        # 에포크 종료 후 평균 손실 계산
        avg_loss = total_loss / num_batches
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

def sample_from_diffusion(model, diffusion, shape, steps=1000, eta=0):
    x = tf.random.normal(shape, dtype=tf.float32)

    for i in reversed(range(steps)):
        print(f"샘플링 단계: {i}/{steps}", end="\r")
        t = tf.ones((shape[0],), dtype=tf.int32) * i

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
            noise = tf.random.normal(shape, dtype=tf.float32)
            x = x + sigma * noise
    
    return x

def load_model(model_path='autoencoder.keras'):
    custom_objects = {
        'Sampling': Sampling
    }
    return keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

def inverse_transform(wavelet_data):
    if wavelet_data.ndim == 4:
        if wavelet_data.shape[3] == 1:
            wavelet_data = wavelet_data[0, :, :, 0]
        else:
            wavelet_data = wavelet_data[0]
    elif wavelet_data.ndim == 3 and wavelet_data.shape[0] == 1:
        wavelet_data = wavelet_data[0]
    samples = []
    try:
        for i in range(wavelet_data.shape[0]):
            try:
                cA = wavelet_data[i, 0]
                cD = wavelet_data[i, 1]
                reconstructed = pywt.idwt(cA, cD, 'db1')
                if reconstructed is not None:
                    if not np.all(np.isnan(reconstructed)) and not np.all(reconstructed == 0):
                        samples.extend(reconstructed) 
            except Exception:
                continue
    except Exception:
        samples = np.zeros(1000)
    samples = np.array(samples)
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
        logger.info("Apple Silicon 감지, Metal 최적화 적용 중...")
        # Metal 성능 향상을 위한 설정
        try:
            # 메모리 제약 완화
            tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
            # 최적화 옵션
            tf.config.optimizer.set_jit(True)  # XLA 컴파일
            # Apple Neural Engine 특화 옵션
            os.environ['TF_METAL_DEVICE_FORCE_CPU'] = '0'  # 항상 Metal 사용
            os.environ['TF_METAL_DEBUG_ERROR_INSERTION'] = '0'  # 디버그 비활성화
            logger.info("Apple Neural Engine 최적화 적용 완료")
        except Exception as e:
            logger.warning(f"Apple Neural Engine 최적화 적용 실패: {str(e)}")
            
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
        steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
        epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 100
        batch_size = int(sys.argv[5]) if len(sys.argv) > 5 else 64
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