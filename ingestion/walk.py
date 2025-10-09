from pathlib import Path
from typing import Iterator, Tuple

def walk_corpus(root_path: Path) -> Iterator[Tuple[Path, Path]]:
    """
    Walk through corpus and yield (file_path, parent_dir) pairs
    parent_dir helps identify Confluence space structure
    """
    root_path = Path(root_path)
    
    if not root_path.exists():
        raise ValueError(f"Root path does not exist: {root_path}")
    
    for file_path in root_path.rglob('*'):
        if not file_path.is_file():
            continue
            
        # Skip hidden files and system files
        if file_path.name.startswith('.'):
            continue
        
        # Skip common non-document files
        skip_extensions = {'.tmp', '.lock', '.DS_Store'}
        if file_path.suffix.lower() in skip_extensions:
            continue
        
        # Get parent directory relative to root
        try:
            parent_dir = file_path.parent.relative_to(root_path)
        except ValueError:
            continue
        
        yield file_path, parent_dir