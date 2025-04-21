import numpy as np
from sentence_transformers import SentenceTransformer

embedding_model = None
embed_dim = 512

def setup_embedding_model():
    global embedding_model
    if embedding_model is not None:
        return embedding_model
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    
    return model

def text_to_embedding(text: str) -> np.ndarray:
    global embedding_model
    
    if embedding_model is None:
        embedding_model = setup_embedding_model()
    embedding = embedding_model.encode([text])
    original_dim = embedding.shape[1]
    if original_dim < embed_dim:
        padding = np.zeros((1, embed_dim - original_dim))
        embedding = np.concatenate([embedding, padding], axis=1)
    elif original_dim > embed_dim:
        embedding = embedding[:, :embed_dim]
    
    return embedding.astype(np.float32)

def adjust_embedding_dimension(embeddings, target_dim=None):
    if target_dim is None:
        target_dim = embed_dim
        
    if len(embeddings.shape) == 1:
        original_dim = embeddings.shape[0]
        if original_dim < target_dim:
            padding = np.zeros(target_dim - original_dim)
            adjusted = np.concatenate([embeddings, padding])
        elif original_dim > target_dim:
            adjusted = embeddings[:target_dim]
        else:
            adjusted = embeddings
    else:
        original_dim = embeddings.shape[1]
        if original_dim < target_dim:
            padding = np.zeros((embeddings.shape[0], target_dim - original_dim))
            adjusted = np.concatenate([embeddings, padding], axis=1)
        elif original_dim > target_dim:
            adjusted = embeddings[:, :target_dim]
        else:
            adjusted = embeddings
            
    return adjusted.astype(np.float32)