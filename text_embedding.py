import numpy as np
from sentence_transformers import SentenceTransformer

embedding_model = None
embed_dim = 512

def setup_embedding_model():
    """텍스트 임베딩 모델 설정"""
    global embedding_model
    if embedding_model is not None:
        return embedding_model
    try:
        # 다국어 지원 텍스트 임베딩 모델
        model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        print("텍스트 임베딩 모델 로드 성공")
    except Exception as e:
        print(f"오류: {e}")
        # 모델 로드 실패 시 더 작은 모델 시도
        try:
            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            print("대체 텍스트 임베딩 모델 로드")
        except:
            # 최후의 방법
            raise ValueError("텍스트 임베딩 모델 로드 실패. sentence-transformers 패키지가 설치되었는지 확인하세요.")
    
    embedding_model = model
    return model

def text_to_embedding(text, target_dim=None):
    """텍스트를 임베딩 벡터로 변환"""
    global embedding_model, embed_dim
    
    if target_dim is None:
        target_dim = embed_dim
    
    if embedding_model is None:
        embedding_model = setup_embedding_model()
    
    # 문자열인지 확인
    if not isinstance(text, str):
        text = str(text)
    
    # 임베딩 생성
    embedding = embedding_model.encode([text])
    
    # 차원 조정
    return adjust_embedding_dimension(embedding, target_dim)

def adjust_embedding_dimension(embeddings, target_dim=None):
    """임베딩 차원 조정"""
    global embed_dim
    
    if target_dim is None:
        target_dim = embed_dim
    
    # 단일 임베딩인 경우
    if len(embeddings.shape) == 1:
        original_dim = embeddings.shape[0]
        if original_dim < target_dim:
            # 패딩 추가
            padding = np.zeros(target_dim - original_dim)
            adjusted = np.concatenate([embeddings, padding])
        elif original_dim > target_dim:
            # 절삭
            adjusted = embeddings[:target_dim]
        else:
            adjusted = embeddings
    else:
        # 배치 임베딩인 경우
        original_dim = embeddings.shape[1]
        if original_dim < target_dim:
            # 패딩 추가
            padding = np.zeros((embeddings.shape[0], target_dim - original_dim))
            adjusted = np.concatenate([embeddings, padding], axis=1)
        elif original_dim > target_dim:
            # 절삭
            adjusted = embeddings[:, :target_dim]
        else:
            adjusted = embeddings
    
    # NaN 값 제거
    adjusted = np.nan_to_num(adjusted)
    return adjusted.astype(np.float32)

def text_to_tokens(text, token_length=256):
    """텍스트를 시퀀스 토큰으로 변환 (웨이블릿 변환에 유용)"""
    embedding = text_to_embedding(text)
    
    # 단일 임베딩을 시퀀스로 변환
    if len(embedding.shape) == 1:
        embedding = np.expand_dims(embedding, axis=0)
    
    # 토큰 길이로 확장 또는 축소
    tokens = np.zeros((embedding.shape[0], token_length, embed_dim))
    
    for i in range(embedding.shape[0]):
        # 임베딩을 반복하여 토큰 길이에 맞춤
        for j in range(token_length):
            tokens[i, j] = embedding[i]
    
    return tokens