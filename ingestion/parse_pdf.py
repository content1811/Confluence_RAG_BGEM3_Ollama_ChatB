from pathlib import Path
from typing import List
from pdfminer.high_level import extract_text
from pdfminer.pdfpage import PDFPage
from core.utils import Chunk, estimate_tokens

def parse_pdf(file_path: Path, max_tokens: int, overlap: int) -> List[Chunk]:
    chunks = []
    
    try:
        # Extract text from entire PDF
        full_text = extract_text(str(file_path))
        
        # Try to extract page by page for better structure
        with open(file_path, 'rb') as f:
            pages = list(PDFPage.get_pages(f))
            
            if len(pages) > 1:
                # Extract per page
                for page_num, page in enumerate(pages, start=1):
                    page_text = extract_text(str(file_path), page_numbers=[page_num-1])
                    
                    if page_text.strip():
                        token_count = estimate_tokens(page_text)
                        
                        # If page is too large, split it
                        if token_count > max_tokens:
                            from core.utils import split_into_chunks
                            sub_chunks = split_into_chunks(page_text, max_tokens, overlap)
                            for i, sub_text in enumerate(sub_chunks):
                                chunks.append(Chunk(
                                    text=sub_text,
                                    section_path=f"page_{page_num}_part_{i}",
                                    token_count=estimate_tokens(sub_text),
                                    extra_meta={"page": page_num}
                                ))
                        else:
                            chunks.append(Chunk(
                                text=page_text,
                                section_path=f"page_{page_num}",
                                token_count=token_count,
                                extra_meta={"page": page_num}
                            ))
            else:
                # Single page PDF
                if full_text.strip():
                    chunks.append(Chunk(
                        text=full_text,
                        section_path="page_1",
                        token_count=estimate_tokens(full_text),
                        extra_meta={"page": 1}
                    ))
    except Exception as e:
        print(f"Error parsing PDF {file_path}: {e}")
    
    return chunks