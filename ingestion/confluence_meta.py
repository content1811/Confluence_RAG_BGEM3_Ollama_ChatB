import json
from pathlib import Path
from typing import Optional, Dict

def extract_confluence_metadata(page_dir: Path) -> Optional[Dict]:
    """Extract metadata from Confluence page.json"""
    page_json = page_dir / "page.json"
    
    if not page_json.exists():
        return None
    
    try:
        with open(page_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = {
            'title': data.get('title', ''),
            'space_key': data.get('space', {}).get('key', ''),
            'version': data.get('version', {}).get('number', 1),
            'updated_at': data.get('version', {}).get('when', ''),
        }
        
        return metadata
    except Exception as e:
        print(f"Error reading page.json from {page_dir}: {e}")
        return None