#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_config
from db.sqlite import SQLiteDB
from embeddings.encoder import EmbeddingEncoder
from embeddings.vector_store import VectorStore
from retrieval.hybrid_search import HybridSearch
from retrieval.reranker import Reranker
from llm.client import LocalLLMClient
from pipeline.query_pipeline import QueryPipeline

def print_response(result: dict):
    """Pretty print query response"""
    print("\n" + "="*80)
    print("ANSWER")
    print("="*80)
    print(f"\n{result['answer']}\n")
    
    # if result.get('citations'):
    #     print("-"*80)
    #     print("SOURCES")
    #     print("-"*80)
    #     for cite in result['citations']:
    #         print(f"[{cite['id']}] {cite['title']}")
    #         print(f"    Section: {cite['section']}")
    #         print(f"    File: {cite['file']}\n")
    
    # print("-"*80)
    # print(f"Confidence: {result['confidence'].upper()}")
    # if result.get('chunks_used'):
    #     print(f"Sources used: {result['chunks_used']}")
    # print("="*80 + "\n")

def main():
    print("\n" + "="*80)
    print("CONFLUENCE RAG CHAT")
    print("="*80 + "\n")
    
    # Load config
    config = load_config()
    print("Loading RAG pipeline...")
    
    # Initialize components
    db = SQLiteDB(config.paths.sqlite_path)
    db.connect()
    
    encoder = EmbeddingEncoder(
        model_name=config.embeddings.model,
        device=config.embeddings.device
    )
    
    vector_store = VectorStore(config.paths.chroma_dir)
    
    hybrid_search = HybridSearch(db, vector_store, encoder, config)
    
    reranker = None
    if config.rerank.enabled:
        reranker = Reranker(
            model_name=config.rerank.model,
            device="cpu"
        )
    
    llm_client = LocalLLMClient(
        base_url=config.llm.base_url,
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens
    )
    
    pipeline = QueryPipeline(db, hybrid_search, reranker, llm_client, config)
    
    # Simple conversation history list
    conversation_history = []
    
    print("✓ Pipeline ready!\n")
    print("Commands:")
    print("  - Type your question and press Enter")
    print("  - Type 'quit' or 'exit' to exit")
    print("  - Type 'clear' to clear screen")
    print("  - Type 'reset' to clear conversation history\n")
    
    while True:
        try:
            question = input("You: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if question.lower() == 'clear':
                import os
                os.system('clear' if os.name != 'nt' else 'cls')
                continue
            
            if question.lower() == 'reset':
                conversation_history = []
                print("\n✓ Conversation history cleared\n")
                continue
            
            print("\nProcessing...\n")
            
            # Query with history
            result = pipeline.query(question, history=conversation_history)
            
            # Add to history
            conversation_history.append({"role": "user", "content": question})
            conversation_history.append({"role": "assistant", "content": result['answer']})
            
            # Keep only last 10 messages (5 turns)
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
            
            print_response(result)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            import traceback
            traceback.print_exc()
    
    db.close()

if __name__ == "__main__":
    main()