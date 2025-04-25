import os
import hashlib
import pickle
import shutil
import concurrent
from concurrent.futures import ThreadPoolExecutor
import glob
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

# 기본 설정 값
sample_size = 2048
WAVELET_TYPE = 'db4'  # 웨이블릿 타입 (db1, db4, sym4 등)
DEFAULT_SAMPLE_RATE = 48000

def check_available_devices():
    """사용 가능한 장치 목록 확인"""
    devices = tf.config.list_physical_devices()
    for device in devices:
        logger.info(f"사용 가능한 장치: {device.name}, 유형: {device.device_type}")
    return devices

def configure_gpu():
    """GPU 설정"""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"GPU: {len(gpus)}개 발견")
            # 메모리 증가 허용
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # 혼합 정밀도 활성화
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("혼합 정밀도(mixed_float16) 활성화됨")
            return True
        else:
            logger.warning("GPU를 찾을 수 없습니다.")
            return False
    except Exception as e:
        logger.error(f"GPU 설정 중 오류 발생: {str(e)}")
        return False

def get_cache_path(filename, cache_dir="wavelet_cache"):
    """파일 경로에 대한 캐시 경로 생성"""
    # 캐시 디렉토리 생성
    os.makedirs(cache_dir, exist_ok=True)
    
    # 파일 이름에서 해시 생성 (전체 경로 사용)
    file_hash = hashlib.md5(filename.encode()).hexdigest()
    
    # 웨이블릿 타입 및 샘플 크기도 캐시 이름에 포함
    params_str = f"{WAVELET_TYPE}_{sample_size}"
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    
    # 캐시 파일명: 파일해시_파라미터해시.pkl
    cache_filename = f"{file_hash}_{params_hash}.pkl"
    return os.path.join(cache_dir, cache_filename)

def save_to_cache(data, cache_path):
    """변환 결과를 캐시에 저장"""
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"캐시 저장 완료: {cache_path}")
        return True
    except Exception as e:
        logger.error(f"캐시 저장 실패: {e}")
        return False

def load_from_cache(cache_path):
    """캐시에서 변환 결과 로드"""
    try:
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"캐시 로드 성공: {cache_path}")
            return data
        return None
    except Exception as e:
        logger.error(f"캐시 로드 실패: {e}")
        return None

def clear_cache(cache_dir="wavelet_cache", older_than_days=None):
    """캐시 정리 함수 (선택적으로 특정 기간보다 오래된 파일만 삭제)"""
    if not os.path.exists(cache_dir):
        return
        
    try:
        if older_than_days is None:
            # 전체 캐시 삭제
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"캐시 디렉토리 초기화 완료: {cache_dir}")
        else:
            # 특정 기간보다 오래된 파일만 삭제
            import time
            current_time = time.time()
            cutoff_time = current_time - (older_than_days * 24 * 3600)
            
            count = 0
            for filename in os.listdir(cache_dir):
                filepath = os.path.join(cache_dir, filename)
                if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff_time:
                    os.remove(filepath)
                    count += 1
            
            logger.info(f"{count}개의 오래된 캐시 파일 삭제 완료")
    except Exception as e:
        logger.error(f"캐시 정리 중 오류 발생: {e}")

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
        self.filepath = path
        
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
        sample_values = []
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
            sample_values.append(float_value)
            
        wave_file.close()
        self.samples = np.array(sample_values, dtype=np.float32)
        logger.info(f'Loaded wave file: {path} (길이: {len(self.samples)})')

    def transform(self) -> list[np.ndarray]:    
        """웨이블릿 변환 병렬 처리"""
        cache_path = get_cache_path(self.filepath)
    
        # 캐시 확인
        cached_result = load_from_cache(cache_path)
        if cached_result is not None:
            logger.info(f"캐시된 웨이블릿 변환 데이터 사용: {self.filepath}")
            return cached_result
        
        result = []
        segments = len(self.samples) // sample_size
        segments_list = []
        
        # 세그먼트 미리 준비 (NumPy 배열로 변환)
        for i in range(segments):
            segment = self.samples[i * sample_size: (i + 1) * sample_size]
            if len(segment) == sample_size:  # 세그먼트가 짧으면 건너뛰기
                # 명시적으로 NumPy 배열로 변환
                segments_list.append(np.array(segment, dtype=np.float32))
        
        def process_segment(segment):
            try:
                # segment가 NumPy 배열인지 확인
                if not isinstance(segment, np.ndarray):
                    segment = np.array(segment, dtype=np.float32)
                    
                # shape 속성 확인
                if not hasattr(segment, 'shape'):
                    return None
                    
                # 노이즈 추가로 증강 (약 10%)
                if np.random.random() < 0.1:
                    noise_level = np.random.uniform(0.001, 0.01)
                    noise = np.random.normal(0, noise_level, segment.shape)
                    segment = segment + noise
                
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
        num_workers = min(32, os.cpu_count() or 24)
        
        # 병렬 처리
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # map 대신 submit/as_completed 사용하여 결과 즉시 처리
            futures = [executor.submit(process_segment, segment) for segment in segments_list]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(segments_list), desc="웨이블릿 변환"):
                res = future.result()
                if res is not None:
                    result.append(res)
        
        save_to_cache(result, cache_path)
        return result

def load_data_parallel(folder: str, use_cache=True, recompute=False):
    """여러 WAV 파일을 병렬로 로드하고 처리 (캐싱 지원)"""
    wav_files = glob.glob(os.path.join(folder, '*.wav'))
    
    # 폴더 전체에 대한 캐시 확인
    folder_hash = hashlib.md5(folder.encode()).hexdigest()
    folder_params = f"{WAVELET_TYPE}_{sample_size}_files{len(wav_files)}"
    folder_params_hash = hashlib.md5(folder_params.encode()).hexdigest()[:8]
    folder_cache_path = os.path.join("wavelet_cache", f"folder_{folder_hash}_{folder_params_hash}.pkl")
    
    # 캐시가 있고 재계산을 요청하지 않았으면 캐시 사용
    if use_cache and not recompute:
        cached_result = load_from_cache(folder_cache_path)
        if cached_result is not None:
            logger.info(f"폴더 전체 캐시 데이터 사용: {folder} (파일 {len(wav_files)}개)")
            return cached_result
    
    logger.info(f"총 {len(wav_files)} 개의 WAV 파일을 처리합니다.")
    data = []
    
    def process_file(wav_file):
        try:
            # 개별 파일 캐시 확인
            if use_cache and not recompute:
                file_cache_path = get_cache_path(wav_file)
                cached_data = load_from_cache(file_cache_path)
                if cached_data is not None:
                    return cached_data
            
            # 캐시가 없거나 재계산 요청이면 계산
            audio_data = Data(wav_file)
            audio_data.filepath = wav_file  # 파일 경로 저장 (캐시 키로 사용)
            result = audio_data.transform()
            
            # 결과 캐싱
            if use_cache:
                file_cache_path = get_cache_path(wav_file)
                save_to_cache(result, file_cache_path)
                
            return result
        except Exception as e:
            logger.error(f"파일 처리 중 오류 발생: {wav_file}, {str(e)}")
            return []
    
    # 병렬 처리
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(process_file, wav_files), total=len(wav_files)))
    
    # 결과 합치기
    for result in results:
        data.extend(result)
    
    # 폴더 전체 결과 캐싱
    if use_cache:
        save_to_cache(np.array(data), folder_cache_path)
    
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
        device = "/device:GPU:0" if tf.config.list_physical_devices('GPU') else "/device:CPU:0"
        
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
    U-Net 모델 구축 함수
    
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

@keras.saving.register_keras_serializable()
class MatMulLayer(keras.layers.Layer):
    def call(self, inputs):
        return tf.matmul(inputs[0], inputs[1])
    
    def compute_output_shape(self, input_shapes):
        return (input_shapes[0][0], input_shapes[0][1], input_shapes[1][2])

@keras.saving.register_keras_serializable()
class ScaleLayer(keras.layers.Layer):
    def __init__(self, scale_factor, **kwargs):
        super().__init__(**kwargs)
        self.scale_factor = scale_factor
    
    def call(self, inputs):
        return inputs / self.scale_factor
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super().get_config()
        config.update({"scale_factor": self.scale_factor})
        return config

def attention_block(x, filters):
    """셀프 어텐션 블록"""
    # 가중치 정규화 추가
    kernel_regularizer = keras.regularizers.l2(1e-5)

    # 컨볼루션 레이어에 적용
    h = keras.layers.Conv2D(
        filters, 3, padding='same',
        kernel_regularizer=kernel_regularizer
    )(x)
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
    
    # Lambda 대신 커스텀 레이어 사용
    attn = MatMulLayer()([k, q])
    attn = ScaleLayer(np.sqrt(filters // 8))(attn)
    attn = keras.layers.Softmax(axis=-1)(attn)
    output = MatMulLayer()([attn, v])
    
    output = keras.layers.Reshape((height, width, filters))(output)
    
    # 원본과 결합
    output = keras.layers.Conv2D(channels, 1, padding='same')(output)
    return keras.layers.Add()([x, output * 0.1])  # 스케일링 적용

# 또는 스펙트럴 정규화 구현
def spectral_norm_conv(x, filters, kernel_size=3):
    # 스펙트럴 정규화는 가중치의 가장 큰 특이값으로 나누어 안정화
    conv_layer = keras.layers.Conv2D(filters, kernel_size, padding='same')
    # 여기서 실제 스펙트럴 정규화를 구현해야 함
    return conv_layer(x)

def train(folder: str, timesteps=1000, epochs=100, batch_size=16, beta_schedule='cosine'):
    """데이터 학습 함수"""
    # 장치 설정
    configure_gpu()
    
    # 데이터 로드
    data = load_data_parallel(folder)
    
    if len(data) == 0:
        logger.error("처리할 데이터가 없습니다. 데이터 로드에 실패했습니다.")
        return
    
    logger.info(f"데이터 형태: {data.shape}")
    
    # 데이터 형태 확인 및 변환
    if data.shape[1] == 2 and data.shape[2] > 1000:
        # 필요한 변환: (samples, 2, width) -> (samples, width, 2)
        data = np.transpose(data, (0, 2, 1))
        logger.info(f"데이터 형태 변환 완료: {data.shape}")

    # 모델 설정
    diffusion = DiffusionModel(timesteps=timesteps, beta_schedule=beta_schedule)
    if len(data.shape) == 3:
        input_shape = (data.shape[1], data.shape[2], 1)
        data = np.expand_dims(data, axis=-1)
    else:
        input_shape = data.shape[1:]
    
    logger.info(f"입력 형태: {input_shape}")
    
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
            time_embedding_dim=256, 
            base_filters=128, 
            depth=5, 
            attention_res=[1, 2, 3, 4]
        )
        
        # 옵티마이저 및 학습률 스케줄러 설정
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-6,
            decay_steps=epochs * (len(data) // batch_size),
            alpha=0.01
        )
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=1e-5,  # L2 정규화 효과
            beta_1=0.9,
            beta_2=0.99,  # 일반적인 0.999보다 작게 설정
            ema_momentum=0.999  # EMA 사용 (더 안정적인 가중치)
        )

        model.compile(optimizer=optimizer, loss='mse')
    
        # 모델 요약 출력
        model.summary()
        
        def train_step_fn(batch_indices):
            batch_data = tf.gather(samples, batch_indices)
            batch_data = tf.cast(batch_data, tf.float16)
            
            accumulated_gradients = [tf.zeros_like(var) for var in model.trainable_variables]
            num_accumulations = 16  # 16번 누적 (가상 배치 크기 = 16 * batch_size)
            grad_norm_clip = 1.0
            total_loss = 0.0
            
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
                
                # 수정: noise 타입을 tf.float16으로 일치시킴
                noise = tf.random.normal(shape=tf.shape(sub_batch), dtype=tf.float16)
                x_noisy = diffusion.q_sample(sub_batch, t, noise)
                t_expanded = tf.expand_dims(t, -1)
                
                with tf.GradientTape() as tape:
                    predicted_noise = model([x_noisy, t_expanded], training=True)
                    
                    # 형태 일치 확인 및 필요시 조정
                    if noise.shape != predicted_noise.shape:
                        noise = tf.reshape(noise, predicted_noise.shape)
                        
                    # 두 텐서의 타입 확인 및 일치
                    if noise.dtype != predicted_noise.dtype:
                        noise = tf.cast(noise, predicted_noise.dtype)
                        
                    loss = tf.reduce_mean(tf.square(noise - predicted_noise)) / num_accumulations
                    if tf.math.is_nan(loss):
                        tf.print("경고: NaN 손실 발생!")
                    elif tf.math.is_inf(loss):
                        tf.print("경고: 무한대 손실 발생!")
                    loss = tf.clip_by_value(loss, 0, 1e6)
                    total_loss += loss * num_accumulations
                
                # 그래디언트 계산 및 누적
                gradients = tape.gradient(loss, model.trainable_variables)
                gradients, _ = tf.clip_by_global_norm(gradients, grad_norm_clip)
                accumulated_gradients = [accum_grad + grad for accum_grad, grad in zip(accumulated_gradients, gradients)]
            optimizer.apply_gradients(zip(accumulated_gradients, model.trainable_variables))
            return total_loss  # 원래 손실 크기로 반환
        
        @tf.function(reduce_retracing=True)
        def train_step(batch_indices):
            per_replica_losses = strategy.run(train_step_fn, args=(batch_indices,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    
    dataset = tf.data.Dataset.range(len(samples))
    dataset = dataset.shuffle(buffer_size=min(len(samples), 10000))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # 학습 시작
    for epoch in range(epochs):
        logger.info(f"에포크 {epoch+1}/{epochs}")

        # 데이터셋을 사용한 학습 루프
        epoch_loss = 0
        step = 0
        
        # 미리 생성한 데이터셋 사용
        for batch_indices in tqdm(dataset, desc=f"Epoch {epoch+1}/{epochs}"):
            loss = train_step(batch_indices)
            epoch_loss += loss
            step += 1
            
            # 메모리 정리 (선택적)
            if step % 5 == 0 and tf.config.list_physical_devices('GPU'):
                tf.keras.backend.clear_session()
                import gc
                gc.collect()
        
        # 에포크 종료 후 평균 손실 계산
        avg_loss = epoch_loss / step
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
    # 모델 입력 형태 확인
    exp_input_shape = model.input[0].shape if isinstance(model.input, list) else model.input.shape
    
    # 배치 크기만 조정하고 나머지 형태는 모델 기대값과 일치시킴
    # 1차원 (배치 크기)만 조정
    batch_shape = (num_samples,)
    # 나머지 차원은 모델 입력 형태에서 가져옴 (None 부분을 제외하고)
    for dim in exp_input_shape[1:]:
        # None이 아닌 차원만 사용
        if dim is not None:
            batch_shape = batch_shape + (dim,)

    logger.info(f"샘플링 배치 형태: {batch_shape}")
    
    if len(batch_shape) <= 1:
        logger.warning("모델 입력 형태를 확인할 수 없습니다. 기본값 사용")
        batch_shape = (num_samples, 1024, 2, 1)

    # 메모리 문제 방지를 위한 더 작은 배치 크기 사용
    x = tf.random.normal(batch_shape, dtype=tf.float32)

    # 생성된 데이터 안정성 보장
    x_np = x.numpy()
    # NaN 또는 무한대 값 처리
    x_np = np.nan_to_num(x_np, nan=0.0, posinf=1.0, neginf=-1.0)
    # 극단값 제한
    x_np = np.clip(x_np, -5.0, 5.0)
    x = tf.convert_to_tensor(x_np, dtype=tf.float32)
    
    # 메모리 사용량 제한
    try:
        # TensorFlow 메모리 사용량 제한
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except Exception as e:
        logger.warning(f"메모리 증가 설정 오류: {e}")

    # 반복적인 노이즈 제거 (메모리 관리 추가)
    for i in tqdm(reversed(range(steps)), desc="샘플링"):
        t = tf.ones((batch_shape[0],), dtype=tf.int32) * i
        
        # 형태 로깅 추가 (처음에만)
        if i == steps-1:
            logger.info(f"샘플링 입력 형태: {x.shape}, 시간 형태: {t[:, None].shape}")
        
        # 세그멘테이션 폴트 방지 - 메모리 해제
        if i % 10 == 0:
            tf.keras.backend.clear_session()
            # 가비지 컬렉션 강제 실행 
            import gc
            gc.collect()
        
        # 예측된 노이즈 계산 (try-except로 오류 처리 강화)
        try:
            predicted_noise = tf.cast(model([x, t[:, None]], training=False), tf.float32)
            
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
                
        except Exception as e:
            logger.error(f"샘플링 스텝 {i}에서 오류 발생: {e}")
            # 오류 발생 시 현재 상태 유지하고 계속 진행
            continue
    
    x_np = x.numpy()
    x_np = np.nan_to_num(x_np, nan=0.0, posinf=1.0, neginf=-1.0)
    x_np = np.clip(x_np, -10.0, 10.0)
    
    # 데이터 유효성 확인 (필요시 로깅)
    x_result = x.numpy()
    has_issue = np.any(np.isnan(x_result)) or np.any(np.isinf(x_result))
    if has_issue:
        logger.warning("생성된 데이터에 문제가 있어 수정합니다.")
        x_result = np.nan_to_num(x_result, nan=0.0, posinf=1.0, neginf=-1.0)
        x_result = np.clip(x_result, -1.0, 1.0)
    
    return tf.convert_to_tensor(x_result, dtype=tf.float32)

def load_model(model_path='diffusion_model.keras'):
    """모델 로드 함수"""
    custom_objects = {
        'Sampling': Sampling,
        'MatMulLayer': MatMulLayer,
        'ScaleLayer': ScaleLayer
    }
    return keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

def inverse_transform(wavelet_data):
    """병렬 웨이블릿 역변환 (스테레오 지원)"""
    original_shape = wavelet_data.shape
    logger.info(f"입력 웨이블릿 데이터 형태: {original_shape}")
    
    # 여러 형태의 입력 처리
    if wavelet_data.ndim == 4:
        if wavelet_data.shape[3] == 1:
            wavelet_data = wavelet_data[:, :, :, 0]  # 채널 차원 제거
    
    # 데이터가 NumPy 배열인지 확인
    if not isinstance(wavelet_data, np.ndarray):
        wavelet_data = np.array(wavelet_data)
    
    if wavelet_data.ndim == 4 and wavelet_data.shape[3] == 1:
        wavelet_data = wavelet_data.squeeze(axis=3)
    
    logger.info(f"처리 전 웨이블릿 데이터 형태: {wavelet_data.shape}")
    
    def process_segment(i):
        try:
            # 인덱스 검사
            if i >= wavelet_data.shape[0]:
                return None
                
            # 데이터 추출 및 형태 확인
            if wavelet_data.ndim == 3:
                max_allowed_dim = 512  # PyWavelets 안전 한계
                
                # 큰 데이터를 작은 청크로 분할 처리
                cA_full = np.array(wavelet_data[i, 0], dtype=np.float32)
                cD_full = np.array(wavelet_data[i, 1], dtype=np.float32)
                
                # 길이 맞추기
                min_len = min(len(cA_full), len(cD_full))
                chunk_size = min(max_allowed_dim, len(cA_full))
                chunk_count = (len(cA_full) + chunk_size - 1) // chunk_size
                
                chunks_reconstructed = []
                for chunk in range(chunk_count):
                    start = chunk * chunk_size
                    end = min(start + chunk_size, len(cA_full))
                    
                    # 청크 크기 확인
                    if end - start < 2:
                        continue  # 너무 작은 청크는 건너뛰기
                    
                    # 해당 청크 계수 추출
                    cA = cA_full[start:end]
                    cD = cD_full[start:end]
                    
                    # NaN 및 무한대 값 대체
                    cA = np.nan_to_num(cA, nan=0.0, posinf=0.1, neginf=-0.1)
                    cD = np.clip(cA, -3.0, 3.0)
                    cD = np.nan_to_num(cD, nan=0.0, posinf=0.1, neginf=-0.1)
                    cD = np.clip(cD, -3.0, 3.0)
                    
                    # 역변환 시도
                    try:
                        chunk_reconstructed = pywt.idwt(cA, cD, WAVELET_TYPE)
                        if chunk_reconstructed is not None:
                            chunks_reconstructed.append(chunk_reconstructed)
                    except Exception as e:
                        logger.warning(f"청크 {chunk} 역변환 실패: {e}")
                
                # 모든 청크 결합
                if chunks_reconstructed:
                    reconstructed = np.concatenate(chunks_reconstructed)
                    reconstructed = np.clip(reconstructed, -1.0, 1.0)
                    return reconstructed
                else:
                    return None
            else:
                # 지원되지 않는 형태
                logger.warning(f"지원되지 않는 데이터 형태: {wavelet_data.shape}")
                return None
        except Exception as e:
            logger.error(f"세그먼트 처리 중 오류: {str(e)}")
            return None
    
    # 병렬 처리
    with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as executor:
        futures = {executor.submit(process_segment, i): i for i in range(wavelet_data.shape[0])}
        
        # 진행 표시줄 준비
        results = [None] * wavelet_data.shape[0]
        for future in tqdm(concurrent.futures.as_completed(futures), 
                         total=wavelet_data.shape[0], 
                         desc="웨이블릿 역변환"):
            idx = futures[future]
            try:
                result = future.result()
                if result is not None:
                    results[idx] = result
            except Exception as e:
                logger.error(f"세그먼트 {idx} 처리 중 예외: {str(e)}")
    
    # None이 아닌 결과들만 연결
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        logger.warning("유효한 오디오 세그먼트가 없습니다!")
        return None
    
    # 스테레오 변환
    stereo_results = []
    for res in valid_results:
        if len(res.shape) == 1:  # 모노인 경우 스테레오로 변환
            stereo = np.zeros((len(res), 2), dtype=np.float32)
            stereo[:, 0] = res  # 왼쪽 채널
            stereo[:, 1] = res  # 오른쪽 채널
            stereo_results.append(stereo)
        else:
            stereo_results.append(res)
    
    # 결과 합치기 (스테레오 형식으로)
    samples = np.concatenate(stereo_results, axis=0)
    logger.info(f"생성된 오디오 형태: {samples.shape}")
    return samples

def save_audio(samples, output_path, sample_rate=44100):
    """오디오 데이터를 WAV 파일로 저장 (스테레오 지원)"""
    samples = normalize_audio(samples)
    samples = np.clip(samples, -1.0, 1.0)
    
    # 스테레오 또는 모노 확인
    is_stereo = len(samples.shape) > 1 and samples.shape[1] == 2
    
    with wave.open(output_path, 'w') as wav_file:
        wav_file.setnchannels(2 if is_stereo else 1)  # 스테레오는 2채널
        wav_file.setsampwidth(2)  # 16-bit 오디오
        wav_file.setframerate(sample_rate)
        
        if is_stereo:
            # 스테레오 처리
            for i in range(len(samples)):
                left = int(samples[i, 0] * 32767.0)
                right = int(samples[i, 1] * 32767.0)
                data = struct.pack('<hh', left, right)
                wav_file.writeframes(data)
        else:
            # 모노 처리 (하위 호환성 유지)
            for sample in samples:
                value = int(sample * 32767.0)
                data = struct.pack('<h', value)
                wav_file.writeframes(data)

def normalize_audio(samples, target_peak=0.9):
    """오디오 샘플 정규화 (스테레오 지원)"""
    # 스테레오인 경우
    if len(samples.shape) > 1 and samples.shape[1] == 2:
        # 각 채널별로 최대 진폭 계산
        max_amp_left = np.max(np.abs(samples[:, 0]))
        max_amp_right = np.max(np.abs(samples[:, 1]))
        max_amp = max(max_amp_left, max_amp_right)
        
        if max_amp > 0:
            gain = target_peak / max_amp
            # 두 채널에 동일한 게인 적용 (스테레오 밸런스 유지)
            samples = samples * gain
    else:
        # 모노인 경우 (기존 로직)
        max_amp = np.max(np.abs(samples))
        if max_amp > 0:
            gain = target_peak / max_amp
            samples = samples * gain
            
    return samples

def generate_audio(output_path='generated.wav', steps=100, eta=0.3, seed=None):
    """오디오 생성 함수"""
    logger.info(f"오디오 생성 시작: {output_path}")
    
    # 랜덤 시드 설정 (재현성)
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)
    
    # 모델 로드
    try:
        model = keras.models.load_model('diffusion_model.keras', compile=False)
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
    shape = (1, 1027, 2, 1)  # 베치 크기, 높이, 너비, 채널
    
    # 출력 형태 미리 확인
    input_details = {'input_shapes': [model.input_shape if isinstance(model.input, list) else [model.input_shape]]}
    logger.info(f"모델 입력 형태: {input_details['input_shapes']}")
    logger.info(f"사용될 생성 형태: {shape}")
    
    # 샘플링
    logger.info(f"샘플링 시작 (스텝: {steps}, eta: {eta})")
    generated = sample_from_diffusion(model, diffusion, shape, steps=steps, eta=eta)
    logger.info("샘플링 완료")
    
    # 웨이블릿 역변환
    logger.info("웨이블릿 역변환 시작")
    try:
        audio_samples = inverse_transform(generated.numpy())
    except Exception as e:
        logger.error(f"역변환 중 오류 발생: {e}")
        raise e
    
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

def generate_from_text(text_description, output_path='generated_from_text.wav', steps=100, eta=0.3):
    """텍스트 설명에서 오디오 생성"""
    # 텍스트-오디오 변환 모듈 가져오기
    try:
        from diffusion_text import generate_audio_seeds
        seeds = generate_audio_seeds(text_description)
        if seeds is not None:
            logger.info(f"텍스트 기반 오디오 생성 시작: '{text_description}'")
            return generate_audio(output_path=output_path, steps=steps, eta=eta, seed=seeds[0], text_description=text_description)
        else:
            logger.error("텍스트 기반 오디오 생성 실패: 유효한 시드 없음")
            return None
    except ImportError:
        logger.warning("text_to_audio 모듈을 가져올 수 없습니다. 텍스트 기반 생성 기능을 사용할 수 없습니다.")
        return None

if __name__ == '__main__':
    # 장치 설정
    configure_gpu()
    check_available_devices()
    
    if len(sys.argv) < 2:
        print("사용법:")
        print("  python main.py train [source_path] [timesteps] [epochs] [batch_size] [beta_schedule]")
        print("  python main.py generate [output_path] [steps] [eta] [seed]")
        print("  python main.py text2audio [text] [output_path] [steps] [eta]")
        sys.exit(1)
    elif sys.argv[1] == 'train':
        source_path = sys.argv[2] if len(sys.argv) > 2 else 'samples'
        steps = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 100
        batch_size = int(sys.argv[5]) if len(sys.argv) > 5 else 256
        beta_schedule = sys.argv[6] if len(sys.argv) > 6 else 'cosine'
        
        logger.info(f"학습 시작: {source_path}, timesteps={steps}, epochs={epochs}, batch_size={batch_size}, beta_schedule={beta_schedule}")
        train(source_path, timesteps=steps, epochs=epochs, batch_size=batch_size, beta_schedule=beta_schedule)
    elif sys.argv[1] == 'generate':
        output_path = sys.argv[2] if len(sys.argv) > 2 else 'generated.wav'
        steps = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        eta = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
        seed = int(sys.argv[5]) if len(sys.argv) > 5 else None
        
        logger.info(f"생성 시작: {output_path}, steps={steps}, eta={eta}, seed={seed}")
        result = generate_audio(output_path=output_path, steps=steps, eta=eta, seed=seed)
        
        if result:
            logger.info(f"생성 완료: {output_path} (샘플 수: {len(result['samples'])})")
        else:
            logger.error("생성 실패")
    elif sys.argv[1] == 'text2audio':
        if len(sys.argv) < 3:
            print("텍스트 설명이 필요합니다.")
            print("사용법: python main.py text2audio [text] [output_path] [steps] [eta]")
            sys.exit(1)
            
        text = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else 'generated_from_text.wav'
        steps = int(sys.argv[4]) if len(sys.argv) > 4 else 50
        eta = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
        
        logger.info(f"텍스트 기반 생성 시작: '{text}'")
        result = generate_from_text(text, output_path=output_path, steps=steps, eta=eta)
        
        if result and 'path' in result:
            logger.info(f"텍스트 기반 생성 완료: {result['path']}")
        else:
            logger.error("텍스트 기반 생성 실패")
    elif sys.argv[1] == 'transform2audio':
        from text_to_audio import generate_audio_from_text
        if len(sys.argv) < 3:
            print("텍스트 설명이 필요합니다.")
            print("사용법: python main.py transform2audio [text] [output_path] [steps] [eta]")
            sys.exit(1)
            
        text = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else 'generated_from_text.wav'
        steps = int(sys.argv[4]) if len(sys.argv) > 4 else 50
        eta = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
        
        logger.info(f"텍스트 기반 생성 시작: '{text}'")
        result = generate_audio_from_text(text, tokenizer_model_path='text_to_audio_model', diffusion_model_path='diffusion_model.keras', output_path=output_path, steps=steps, eta=eta)
        
        if result:
            logger.info(f"텍스트 기반 생성 완료: {result['path']}")
        else:
            logger.error("텍스트 기반 생성 실패")
    else:
        print(f"알 수 없는 명령: {sys.argv[1]}")
        print("사용법:")
        print("  python main.py train [source_path] [timesteps] [epochs] [batch_size] [beta_schedule]")
        print("  python main.py generate [output_path] [steps] [eta] [seed]")
        print("  python main.py text2audio [text] [output_path] [steps] [eta]")
        sys.exit(1)