from sentence_transformers import SentenceTransformer
import numpy as np

# Global model instance (loaded once)
_model = None

def get_model():
    """Lazy-load the SBERT model"""
    global _model
    if _model is None:
        print("[Featurizer] Loading SBERT model 'all-MiniLM-L6-v2'...")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        print("[Featurizer] Model loaded successfully.")
    return _model

def embed(text: str) -> np.ndarray:
    """
    Converts text into a feature vector using SBERT.
    
    Args:
        text: The text to embed (combined_context from a node)
        
    Returns:
        A numpy array representing the embedding vector
    """
    if not text or len(text.strip()) == 0:
        print("[Featurizer] WARNING: Empty text provided, returning zero vector")
        model = get_model()
        return np.zeros(model.get_sentence_embedding_dimension())
    
    model = get_model()
    
    # Truncate if too long (SBERT has token limits)
    max_length = 5000  # characters, roughly ~1000 tokens
    if len(text) > max_length:
        text = text[:max_length]
    
    try:
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding
    except Exception as e:
        print(f"[Featurizer] ERROR: Failed to embed text: {e}")
        return np.zeros(model.get_sentence_embedding_dimension())

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes cosine similarity between two vectors.
    
    Args:
        vec1: First embedding vector
        vec2: Second embedding vector
        
    Returns:
        Cosine similarity score (0.0 to 1.0)
    """
    try:
        # Handle zero vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        
        # Clamp to [0, 1] range (cosine can be negative, but we want positive weights)
        similarity = max(0.0, min(1.0, similarity))
        
        return float(similarity)
    except Exception as e:
        print(f"[Featurizer] ERROR: Failed to compute cosine similarity: {e}")
        return 0.0


# --- Main execution block for testing ---
if __name__ == "__main__":
    print("--- Running Featurizer Test ---\n")
    
    test_texts = [
        "Nvidia is a technology company focused on GPUs and AI hardware.",
        "Nvidia produces graphics processing units for gaming and artificial intelligence.",
        "Apple makes consumer electronics like iPhones and MacBooks.",
    ]
    
    print("Computing embeddings...\n")
    embeddings = [embed(text) for text in test_texts]
    
    print(f"Embedding dimension: {embeddings[0].shape[0]}\n")
    
    print("Computing pairwise similarities:")
    print(f"Text 1 vs Text 2 (similar): {cosine_similarity(embeddings[0], embeddings[1]):.4f}")
    print(f"Text 1 vs Text 3 (different): {cosine_similarity(embeddings[0], embeddings[2]):.4f}")
    print(f"Text 2 vs Text 3 (different): {cosine_similarity(embeddings[1], embeddings[2]):.4f}")
    
    print("\n--- Test Complete ---")

