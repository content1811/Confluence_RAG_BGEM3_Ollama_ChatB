from sentence_transformers import CrossEncoder
from typing import List, Tuple, Dict

class Reranker:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model = CrossEncoder(model_name, device=device)
        print(f"✓ Loaded reranker model on {device}")
    
    def rerank(self, query: str, chunks: List[Dict], top_n: int) -> List[Tuple[Dict, float]]:
        """
        Rerank chunks using cross-encoder
        Returns: List of (chunk_dict, score) tuples
        """
        if not chunks or len(chunks) == 0:
            return []
        
        try:
            # Prepare pairs
            pairs = [(query, chunk.get('text', '')) for chunk in chunks]
            
            # Get reranking scores
            scores = self.model.predict(pairs, show_progress_bar=False)
            
            # Combine chunks with scores
            ranked = [(chunk, float(score)) for chunk, score in zip(chunks, scores)]
            
            # Sort by score descending and take top_n
            ranked.sort(key=lambda x: x[1], reverse=True)
            
            return ranked[:top_n]
            
        except Exception as e:
            print(f"❌ Reranking error: {e}")
            import traceback
            traceback.print_exc()
            # Return original chunks with default scores
            return [(chunk, 0.5) for chunk in chunks[:top_n]]

