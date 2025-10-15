from typing import List, Dict

SYSTEM_PROMPT = """You are a helpful AI assistant with access to company documentation. You have two modes:

1) DOC-GROUNDED: When relevant documentation context is provided, answer based on it and cite sources.
2) GENERAL: When no documentation is available, respond naturally using your general knowledge. Be helpful and conversational.

Rules:
- When using documentation, cite sources with [Doc: filename/page]
- When no documentation is available, respond naturally as a helpful assistant
- Never claim you can't answer simple questions or greetings
- Be conversational and friendly
- For questions about company-specific information without documentation, say "I don't have that information in the documentation"
"""

def build_user_prompt(question: str, context: str, mode_hint: str, history: str = "") -> str:
    """
    Build a complete user prompt with question, context, mode, and history
    
    Args:
        question: User's question
        context: Retrieved context (formatted chunks)
        mode_hint: Mode hint (DOC-GROUNDED or GENERAL)
        history: Formatted conversation history
    
    Returns:
        Complete prompt string
    """
    prompt_parts = []
    
    if history:
        prompt_parts.append(f"Previous conversation:\n{history}\n")
    
    prompt_parts.append(f"Question: {question}\n")
    prompt_parts.append(f"Mode: {mode_hint}\n")
    
    if context:
        prompt_parts.append(f"Documentation Context:\n{context}\n")
    
    if mode_hint == "DOC-GROUNDED":
        prompt_parts.append("\nInstructions: Answer based on the documentation context. Cite sources.")
    else:  # GENERAL
        prompt_parts.append("\nInstructions: No documentation available. Respond naturally and helpfully as an AI assistant.")
    
    prompt_parts.append("\nAnswer:")
    
    return "\n".join(prompt_parts)

def format_context(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into context string for LLM
    
    Args:
        chunks: List of chunk dicts with 'text' and 'citation' keys
    
    Returns:
        Formatted context string with source labels
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        citation = chunk.get('citation', {})
        title = citation.get('title', 'Unknown')
        text = chunk.get('text', '')
        context_parts.append(f"[Source {i} - {title}]\n{text}")
    
    return "\n\n---\n\n".join(context_parts)

def format_citations(chunks: List[Dict]) -> List[Dict]:
    """
    Extract and format citations from chunks for API response
    
    Args:
        chunks: List of chunk dicts with 'citation' and 'section_path' keys
    
    Returns:
        List of citation dicts with id, title, section, and file
    """
    citations = []
    for i, chunk in enumerate(chunks, 1):
        citation = chunk.get('citation', {})
        citations.append({
            'id': i,
            'title': citation.get('title', 'Unknown'),
            'section': chunk.get('section_path', 'Unknown'),
            'file': citation.get('relpath', 'Unknown')
        })
    
    return citations
