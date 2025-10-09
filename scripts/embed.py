#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_config
from db.sqlite import SQLiteDB
from embeddings.encoder import EmbeddingEncoder
from embeddings.vector_store import VectorStore
from tqdm import tqdm

def main():
    config = load_config()
    
    print("=== Phase 2: Embedding Generation ===\n")
    
    # Initialize components
    db = SQLiteDB(config.paths.sqlite_path)
    db.connect()
    
    encoder = EmbeddingEncoder(
        model_name=config.embeddings.model,
        device=config.embeddings.device
    )
    
    vector_store = VectorStore(config.paths.chroma_dir)
    
    # Check if embeddings already exist
    existing_count = vector_store.get_count()
    if existing_count > 0:
        print(f"⚠ Found {existing_count} existing embeddings")
        response = input("Delete and rebuild? (y/n): ")
        if response.lower() == 'y':
            vector_store.delete_collection()
            vector_store = VectorStore(config.paths.chroma_dir)
        else:
            print("Keeping existing embeddings")
            db.close()
            return
    
    # Fetch all chunks from database
    cursor = db.conn.execute("SELECT chunk_id, text, doc_id FROM chunks ORDER BY chunk_id")
    all_chunks = cursor.fetchall()
    
    if not all_chunks:
        print("No chunks found in database. Run Phase 1 first.")
        db.close()
        return
    
    print(f"\nProcessing {len(all_chunks)} chunks...")
    
    # Process in batches
    batch_size = config.embeddings.batch_size
    
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding batches"):
        batch = all_chunks[i:i + batch_size]
        
        chunk_ids = [str(row[0]) for row in batch]
        texts = [row[1] for row in batch]
        doc_ids = [row[2] for row in batch]
        
        # Generate embeddings
        embeddings = encoder.encode_batch(texts, batch_size=batch_size)
        
        # Prepare metadata
        metadatas = [{"doc_id": doc_id} for doc_id in doc_ids]
        
        # Store in vector database
        vector_store.add_embeddings(chunk_ids, embeddings, metadatas, texts)
    
    final_count = vector_store.get_count()
    print(f"\n✓ Embedded {final_count} chunks")
    print(f"✓ Vector dimension: {encoder.dimension}")
    
    db.close()
    print("\n✓ Phase 2 complete!")

if __name__ == "__main__":
    main()