import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

class VectorStore:
    def __init__(self, chroma_dir: str, collection_name: str = "chunks"):
        Path(chroma_dir).mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": 32,
                "hnsw:construction_ef": 200,
                "hnsw:search_ef": 256
            }
        )
        print(f"✓ Vector store initialized at {chroma_dir}")
    
    def add_embeddings(self, chunk_ids: List[str], embeddings: np.ndarray, 
                       metadatas: List[Dict], texts: List[str]):
        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=texts
        )
    
    def search(self, query_embedding: np.ndarray, k: int = 50) -> Tuple[List[int], List[float]]:
        """
        Search for similar embeddings
        Returns: (chunk_ids, distances)
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )
        
        # FIXED: Proper empty check to avoid numpy boolean ambiguity
        if not results or 'ids' not in results:
            return [], []
        
        ids_list = results.get('ids', [])
        if not ids_list or len(ids_list) == 0 or len(ids_list[0]) == 0:
            return [], []
        
        chunk_ids = [int(cid) for cid in ids_list[0]]
        distances = results.get('distances', [[]])[0]
        
        return chunk_ids, distances
    
    def get_count(self) -> int:
        return self.collection.count()
    
    def delete_collection(self):
        self.client.delete_collection(self.collection.name)