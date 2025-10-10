from typing import List, Dict, Tuple
from collections import defaultdict
import math

class HybridSearch:
    def __init__(self, db, vector_store, encoder, config):
        self.db = db
        self.vector_store = vector_store
        self.encoder = encoder
        self.config = config
    
    def search(self, query: str) -> List[Tuple[int, float]]:
        """
        Perform hybrid search combining semantic and keyword search
        Returns: List of (chunk_id, score) tuples
        """
        # Semantic search
        query_embedding = self.encoder.encode_single(query)
        vec_chunk_ids, vec_distances = self.vector_store.search(
            query_embedding, 
            k=self.config.retrieval.k_vec
        )
        
        # Convert distances to similarity scores (cosine similarity)
        vec_scores = {cid: 1 - dist for cid, dist in zip(vec_chunk_ids, vec_distances)}
        
        # Keyword search (FTS) - handle errors gracefully
        try:
            fts_results = self.db.fts_search(query, limit=self.config.retrieval.k_fts)
            fts_scores = self._normalize_bm25_scores(fts_results)
        except Exception as e:
            print(f"Warning: FTS search failed ({e}), using semantic search only")
            fts_scores = {}
        
        # If FTS failed, return semantic results only
        if not fts_scores:
            return sorted(vec_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Fusion
        if self.config.retrieval.fusion == "rrf":
            combined = self._reciprocal_rank_fusion(vec_scores, fts_scores)
        else:
            combined = self._weighted_fusion(vec_scores, fts_scores)
        
        # Sort by score descending
        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return sorted_results
    
    def _normalize_bm25_scores(self, fts_results: List[Tuple[int, float]]) -> Dict[int, float]:
        """Normalize BM25 scores to 0-1 range"""
        if not fts_results:
            return {}
        
        scores = [score for _, score in fts_results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return {cid: 1.0 for cid, _ in fts_results}
        
        normalized = {}
        for chunk_id, score in fts_results:
            norm_score = (score - min_score) / (max_score - min_score)
            normalized[chunk_id] = norm_score
        
        return normalized
    
    def _reciprocal_rank_fusion(self, vec_scores: Dict, fts_scores: Dict, k: int = 60) -> Dict:
        """Reciprocal Rank Fusion"""
        combined = defaultdict(float)
        
        # Sort by score to get ranks
        vec_ranked = sorted(vec_scores.items(), key=lambda x: x[1], reverse=True)
        fts_ranked = sorted(fts_scores.items(), key=lambda x: x[1], reverse=True)
        
        # RRF formula: 1 / (k + rank)
        for rank, (chunk_id, _) in enumerate(vec_ranked, start=1):
            combined[chunk_id] += 1 / (k + rank)
        
        for rank, (chunk_id, _) in enumerate(fts_ranked, start=1):
            combined[chunk_id] += 1 / (k + rank)
        
        return dict(combined)
    
    def _weighted_fusion(self, vec_scores: Dict, fts_scores: Dict) -> Dict:
        """Weighted score fusion"""
        combined = defaultdict(float)
        
        w_semantic = self.config.retrieval.weights['semantic']
        w_keyword = self.config.retrieval.weights['keyword']
        
        for chunk_id, score in vec_scores.items():
            combined[chunk_id] += w_semantic * score
        
        for chunk_id, score in fts_scores.items():
            combined[chunk_id] += w_keyword * score
        
        return dict(combined)