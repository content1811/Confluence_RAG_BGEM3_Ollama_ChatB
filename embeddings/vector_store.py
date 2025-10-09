import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Optional
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
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✓ Vector store initialized at {chroma_dir}")
    
    def add_embeddings(self, chunk_ids: List[str], embeddings: np.ndarray, 
                       metadatas: List[Dict], texts: List[str]):
        """Add embeddings to the collection"""
        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=texts
        )
    
    def search(self, query_embedding: np.ndarray, k: int = 50) -> tuple:
        """
        Search for similar embeddings
        Returns: (chunk_ids, distances)
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )
        
        if not results['ids'] or not results['ids'][0]:
            return [], []
        
        chunk_ids = [int(cid) for cid in results['ids'][0]]
        distances = results['distances'][0]
        
        return chunk_ids, distances
    
    def get_count(self) -> int:
        """Get number of vectors in collection"""
        return self.collection.count()
    
    def delete_collection(self):
        """Delete the collection"""
        self.client.delete_collection(self.collection.name)