import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_config
from db.sqlite import SQLiteDB
from embeddings.encoder import EmbeddingEncoder
from embeddings.vector_store import VectorStore
from retrieval.hybrid_search import HybridSearch
from retrieval.reranker import Reranker

def main():
    config = load_config()
    
    print("=== Phase 2: Search Test ===\n")
    
    # Initialize components
    db = SQLiteDB(config.paths.sqlite_path)
    db.connect()
    
    encoder = EmbeddingEncoder(
        model_name=config.embeddings.model,
        device=config.embeddings.device
    )
    
    vector_store = VectorStore(config.paths.chroma_dir)
    
    hybrid_search = HybridSearch(db, vector_store, encoder, config)
    
    # Optional reranker
    reranker = None
    if config.rerank.enabled:
        reranker = Reranker(
            model_name=config.rerank.model,
            device="cpu"
        )
    
    print("Ready for queries!\n")
    
    while True:
        query = input("\nEnter query (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        # Perform hybrid search
        results = hybrid_search.search(query)
        
        if not results:
            print("No results found.")
            continue
        
        print(f"\nFound {len(results)} results")
        
        # Get top chunks
        top_k = min(20, len(results))
        top_chunk_ids = [cid for cid, _ in results[:top_k]]
        
        # Fetch chunk details
        chunks = []
        for chunk_id in top_chunk_ids:
            chunk = db.get_chunk(chunk_id)
            if chunk:
                chunks.append(chunk)
        
        # Rerank if enabled
        if reranker and chunks:
            print("Reranking...")
            reranked = reranker.rerank(query, chunks, config.rerank.top_n)
            
            print(f"\n{'='*80}")
            print(f"Top {len(reranked)} results after reranking:")
            print(f"{'='*80}\n")
            
            for idx, (chunk, score) in enumerate(reranked, 1):
                citation = db.get_citation(chunk['chunk_id'])
                
                print(f"[{idx}] Score: {score:.4f}")
                print(f"    Document: {citation['title']}")
                print(f"    Section: {chunk['section_path']}")
                print(f"    Text: {chunk['text'][:200]}...")
                print()
        else:
            print(f"\n{'='*80}")
            print(f"Top 10 results:")
            print(f"{'='*80}\n")
            
            for idx, chunk in enumerate(chunks[:10], 1):
                citation = db.get_citation(chunk['chunk_id'])
                
                print(f"[{idx}]")
                print(f"    Document: {citation['title']}")
                print(f"    Section: {chunk['section_path']}")
                print(f"    Text: {chunk['text'][:200]}...")
                print()
    
    db.close()
    print("\nGoodbye!")

if __name__ == "__main__":
    main()