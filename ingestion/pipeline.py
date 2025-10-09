from pathlib import Path
from typing import List, Optional
from core.utils import Chunk, compute_sha256, detect_file_type, redact_secrets
from core.config import Config
from ingestion.walk import walk_corpus
from ingestion.confluence_meta import extract_confluence_metadata
from ingestion.parse_html import parse_html
from ingestion.parse_pdf import parse_pdf
from ingestion.parse_pptx import parse_pptx
from ingestion.parse_xlsx import parse_xlsx

class IngestionPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.parsers = {
            'html': parse_html,
            'xhtml': parse_html,
            'pdf': parse_pdf,
            'pptx': parse_pptx,
            'xlsx': parse_xlsx,
        }
    
    def process_file(self, file_path: Path, parent_dir: Path) -> Optional[tuple]:
        """
        Process a single file and return (metadata, chunks)
        Returns None if file should be skipped
        """
        # Compute file hash
        sha256 = compute_sha256(file_path)
        file_type = detect_file_type(file_path)
        
        if file_type == 'unknown':
            return None
        
        # Get relative path from root
        relpath = str(file_path.relative_to(self.config.paths.scrape_root))
        
        # Extract Confluence metadata if available
        confluence_meta = extract_confluence_metadata(file_path.parent)
        
        title = confluence_meta.get('title') if confluence_meta else file_path.stem
        space_key = confluence_meta.get('space_key') if confluence_meta else None
        version = confluence_meta.get('version') if confluence_meta else 1
        updated_at = confluence_meta.get('updated_at') if confluence_meta else None
        
        # Parse file into chunks
        chunks = self._parse_file(file_path, file_type)
        
        if not chunks:
            return None
        
        # Apply redaction if enabled
        if self.config.redaction.enabled:
            chunks = self._redact_chunks(chunks)
        
        metadata = {
            'relpath': relpath,
            'title': title,
            'space_key': space_key,
            'version': version,
            'file_type': file_type,
            'updated_at': updated_at,
            'sha256': sha256
        }
        
        return metadata, chunks
    
    def _parse_file(self, file_path: Path, file_type: str) -> List[Chunk]:
        """Parse file based on type"""
        parser = self.parsers.get(file_type)
        
        if not parser:
            return []
        
        try:
            return parser(
                file_path,
                self.config.chunking.max_tokens,
                self.config.chunking.overlap_tokens
            )
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return []
    
    def _redact_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Apply redaction to sensitive patterns"""
        redacted_chunks = []
        for chunk in chunks:
            redacted_text = redact_secrets(chunk.text, self.config.redaction.patterns)
            redacted_chunks.append(Chunk(
                text=redacted_text,
                section_path=chunk.section_path,
                token_count=chunk.token_count,
                extra_meta=chunk.extra_meta
            ))
        return redacted_chunks
    
    def run(self, db):
        """Run ingestion pipeline"""
        processed = 0
        skipped = 0
        
        print(f"Starting ingestion from {self.config.paths.scrape_root}")
        
        for file_path, parent_dir in walk_corpus(self.config.paths.scrape_root):
            result = self.process_file(file_path, parent_dir)
            
            if result is None:
                skipped += 1
                continue
            
            metadata, chunks = result
            
            # Check if already processed (by hash)
            if db.document_exists(metadata['sha256']):
                skipped += 1
                continue
            
            # Insert document
            doc_id = db.insert_document(
                relpath=metadata['relpath'],
                title=metadata['title'],
                space_key=metadata['space_key'],
                version=metadata['version'],
                file_type=metadata['file_type'],
                updated_at=metadata['updated_at'],
                sha256=metadata['sha256']
            )
            
            # Insert chunks
            for chunk in chunks:
                db.insert_chunk(
                    doc_id=doc_id,
                    section_path=chunk.section_path,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    extra_meta=chunk.extra_meta
                )
            
            processed += 1
            if processed % 10 == 0:
                print(f"Processed: {processed}, Skipped: {skipped}")
        
        print(f"\nIngestion complete. Processed: {processed}, Skipped: {skipped}")