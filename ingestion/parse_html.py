from bs4 import BeautifulSoup
from pathlib import Path
from typing import List
from core.utils import Chunk, estimate_tokens, split_into_chunks

def parse_html(file_path: Path, max_tokens: int, overlap: int) -> List[Chunk]:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'lxml')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    chunks = []
    
    # Try to extract structured content with headings
    sections = extract_sections(soup)
    
    if sections:
        for section_path, section_text in sections:
            section_chunks = split_into_chunks(section_text, max_tokens, overlap)
            for i, chunk_text in enumerate(section_chunks):
                chunks.append(Chunk(
                    text=chunk_text,
                    section_path=f"{section_path}[{i}]" if len(section_chunks) > 1 else section_path,
                    token_count=estimate_tokens(chunk_text)
                ))
    else:
        # Fallback: treat as plain text
        text = soup.get_text(separator=' ', strip=True)
        text_chunks = split_into_chunks(text, max_tokens, overlap)
        for i, chunk_text in enumerate(text_chunks):
            chunks.append(Chunk(
                text=chunk_text,
                section_path=f"section_{i}",
                token_count=estimate_tokens(chunk_text)
            ))
    
    return chunks

def extract_sections(soup: BeautifulSoup) -> List[tuple]:
    """Extract content organized by headings"""
    sections = []
    current_path = []
    current_text = []
    
    for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'div', 'li', 'td']):
        if element.name in ['h1', 'h2', 'h3', 'h4']:
            # Save previous section
            if current_text:
                path_str = ' > '.join(current_path) if current_path else 'root'
                sections.append((path_str, ' '.join(current_text)))
                current_text = []
            
            # Update path
            level = int(element.name[1])
            heading_text = element.get_text(strip=True)
            
            # Adjust path based on heading level
            if level <= len(current_path):
                current_path = current_path[:level-1]
            current_path.append(heading_text)
        else:
            text = element.get_text(strip=True)
            if text:
                current_text.append(text)
    
    # Save last section
    if current_text:
        path_str = ' > '.join(current_path) if current_path else 'root'
        sections.append((path_str, ' '.join(current_text)))
    
    return sections