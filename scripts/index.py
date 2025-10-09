#!/usr/bin/env python3
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_config
from db.sqlite import SQLiteDB
from ingestion.pipeline import IngestionPipeline

def main():
    # Load configuration
    config = load_config()
    
    print("=== Phase 1: Ingestion & Normalization ===\n")
    
    # Initialize database
    print(f"Initializing database at {config.paths.sqlite_path}")
    db = SQLiteDB(config.paths.sqlite_path)
    db.connect()
    db.init_schema()
    print("✓ Database initialized\n")
    
    # Run ingestion
    pipeline = IngestionPipeline(config)
    pipeline.run(db)
    
    # Close database
    db.close()
    print("\n✓ Phase 1 complete!")

if __name__ == "__main__":
    main()