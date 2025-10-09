import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class PathsConfig:
    scrape_root: str
    sqlite_path: str
    chroma_dir: str

@dataclass
class EmbeddingsConfig:
    model: str
    dim: int
    batch_size: int
    device: str

@dataclass
class ChunkingConfig:
    max_tokens: int
    overlap_tokens: int
    prefer_structure: bool

@dataclass
class RetrievalConfig:
    k_vec: int
    k_fts: int
    fusion: str
    weights: Dict[str, float]

@dataclass
class RerankConfig:
    enabled: bool
    model: str
    top_n: int

@dataclass
class AbstainConfig:
    min_rerank_score: float
    min_chunks: int

@dataclass
class LLMConfig:
    model: str
    base_url: str
    temperature: float
    max_tokens: int

@dataclass
class RedactionConfig:
    enabled: bool
    patterns: List[str]

@dataclass
class Config:
    paths: PathsConfig
    embeddings: EmbeddingsConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    rerank: RerankConfig
    abstain: AbstainConfig
    llm: LLMConfig
    redaction: RedactionConfig

def load_config(config_path: str = "configs/rag.yaml") -> Config:
    with open(config_path) as f:
        data = yaml.safe_load(f)
    
    return Config(
        paths=PathsConfig(**data["paths"]),
        embeddings=EmbeddingsConfig(**data["embeddings"]),
        chunking=ChunkingConfig(**data["chunking"]),
        retrieval=RetrievalConfig(**data["retrieval"]),
        rerank=RerankConfig(**data["rerank"]),
        abstain=AbstainConfig(**data["abstain"]),
        llm=LLMConfig(**data["llm"]),
        redaction=RedactionConfig(**data["redaction"])
    )