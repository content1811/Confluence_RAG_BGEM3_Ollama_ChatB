from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np

class HybridSearch:
    def __init__(self, db, vector_store, encoder, config):
        self.db = db
        self.vector_store = vector_store
        self.encoder = encoder
        self.config = config
    
    def search(self, query: str) -> List[Tuple[int, float]]:
        try:
            qlen = len(query.split())
            k_dense = 30 if qlen <= 4 else 20
            k_fts = 30 if qlen <= 4 else 20
            
            query_embedding = self.encoder.encode_single(query)
            vec_chunk_ids, vec_distances = self.vector_store.search(
                query_embedding, 
                k=k_dense
            )
            
            # Safely create vec_scores dict
            vec_scores = {}
            if vec_chunk_ids is not None and len(vec_chunk_ids) > 0:
                vec_scores = {cid: 1 - dist for cid, dist in zip(vec_chunk_ids, vec_distances)}
            
            # Apply MMR for diversity
            if vec_chunk_ids is not None and len(vec_chunk_ids) > 0:
                vec_scores = self._apply_mmr(vec_chunk_ids, vec_distances, query_embedding, k_dense)
            
            try:
                fts_results = self.db.fts_search(query, limit=k_fts)
                fts_scores = self._normalize_bm25_scores(fts_results)
            except Exception as e:
                print(f"Warning: FTS search failed ({e}), using semantic search only")
                fts_scores = {}
            
            # If no FTS results, return semantic only
            if not fts_scores:
                return sorted(vec_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Combine results
            combined = self._reciprocal_rank_fusion(vec_scores, fts_scores)
            return sorted(combined.items(), key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _apply_mmr(self, chunk_ids: List[int], distances: List[float], 
                   query_emb: np.ndarray, k: int, lambda_mult: float = 0.5) -> Dict[int, float]:
        try:
            embeddings = {}
            for cid in chunk_ids:
                try:
                    emb = self.vector_store.collection.get(ids=[str(cid)], include=['embeddings'])
                    # FIXED: Proper check without numpy boolean ambiguity
                    if emb and 'embeddings' in emb and emb['embeddings'] and len(emb['embeddings']) > 0:
                        embedding_list = emb['embeddings'][0]
                        if embedding_list is not None and len(embedding_list) > 0:
                            embeddings[cid] = np.array(embedding_list)
                except Exception:
                    # Silently skip - these warnings are expected for some chunks
                    continue
            
            if not embeddings:
                return {cid: 1 - dist for cid, dist in zip(chunk_ids, distances)}
            
            chosen, chosen_ids = [], set()
            cand_ids = list(embeddings.keys())
            cand_embs = np.stack([embeddings[cid] for cid in cand_ids])
            q = query_emb / (np.linalg.norm(query_emb) + 1e-12)
            sims = cand_embs @ q
            
            for _ in range(min(k, len(cand_ids))):
                if not chosen:
                    idx = int(np.argmax(sims))
                    chosen.append(cand_ids[idx])
                    chosen_ids.add(cand_ids[idx])
                    continue
                
                chosen_embs = np.stack([embeddings[c] for c in chosen])
                redundancy = cand_embs @ chosen_embs.T
                max_red = redundancy.max(axis=1)
                mmr_score = lambda_mult * sims - (1 - lambda_mult) * max_red
                
                for _ in range(len(mmr_score)):
                    idx = int(np.argmax(mmr_score))
                    if cand_ids[idx] not in chosen_ids:
                        chosen.append(cand_ids[idx])
                        chosen_ids.add(cand_ids[idx])
                        break
                    mmr_score[idx] = -1e9
            
            return {cid: 0.9 - (i * 0.05) for i, cid in enumerate(chosen)}
            
        except Exception as e:
            print(f"Warning: MMR failed ({e}), using simple scores")
            return {cid: 1 - dist for cid, dist in zip(chunk_ids, distances)}
    
    def _normalize_bm25_scores(self, fts_results: List[Tuple[int, float]]) -> Dict[int, float]:
        if not fts_results or len(fts_results) == 0:
            return {}
        
        scores = [score for _, score in fts_results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return {cid: 1.0 for cid, _ in fts_results}
        
        return {cid: (score - min_score) / (max_score - min_score) 
                for cid, score in fts_results}
    
    def _reciprocal_rank_fusion(self, vec_scores: Dict, fts_scores: Dict, k: int = 60) -> Dict:
        combined = defaultdict(float)
        
        vec_ranked = sorted(vec_scores.items(), key=lambda x: x[1], reverse=True)
        fts_ranked = sorted(fts_scores.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (chunk_id, _) in enumerate(vec_ranked, start=1):
            combined[chunk_id] += 1 / (k + rank)
        
        for rank, (chunk_id, _) in enumerate(fts_ranked, start=1):
            combined[chunk_id] += 1 / (k + rank)
        
        return dict(combined)

