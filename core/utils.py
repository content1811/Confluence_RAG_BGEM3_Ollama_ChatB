import hashlib
import mimetypes
import re
from pathlib import Path
from typing import List
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    section_path: str
    token_count: int
    extra_meta: dict = None

def compute_sha256(file_path: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def detect_file_type(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    
    type_map = {
        '.html': 'html',
        '.xhtml': 'xhtml',
        '.pdf': 'pdf',
        '.pptx': 'pptx',
        '.xlsx': 'xlsx',
        '.jpg': 'jpg',
        '.jpeg': 'jpg',
        '.png': 'png',
        '.json': 'json'
    }
    
    return type_map.get(suffix, 'unknown')

def estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ~= 4 chars"""
    return len(text) // 4

def redact_secrets(text: str, patterns: List[str]) -> str:
    """Redact sensitive patterns from text"""
    for pattern in patterns:
        text = re.sub(
            f'({pattern})[\s:=]+[^\s]+',
            r'\1: [REDACTED]',
            text,
            flags=re.IGNORECASE
        )
    return text

def split_into_chunks(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    """Simple chunking by token count with overlap"""
    words = text.split()
    chunks = []
    
    chars_per_token = 4
    approx_words_per_chunk = (max_tokens * chars_per_token) // 5
    overlap_words = (overlap_tokens * chars_per_token) // 5
    
    start = 0
    while start < len(words):
        end = start + approx_words_per_chunk
        chunk_words = words[start:end]
        chunk_text = ' '.join(chunk_words)
        
        if chunk_text.strip():
            chunks.append(chunk_text)
        
        start = end - overlap_words
        if start < 0:
            start = end
            
    return chunks