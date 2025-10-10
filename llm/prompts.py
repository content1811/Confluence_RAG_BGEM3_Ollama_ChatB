from typing import List, Dict

SYSTEM_PROMPT = """You are a helpful assistant for internal documentation. Answer questions naturally and conversationally based on the provided context.

CRITICAL RULES:
1. Answer directly and naturally - do NOT explain your reasoning process
2. Do NOT mention "the documents", "the sources", or "the context provided"
3. Do NOT say things like "Based on the provided information" or "Looking at the sources"
4. Answer as if you naturally know this information
5. Be concise and direct
6. If you don't have enough information, simply say: "This information is not available in our documentation."
7. Always cite sources at the end using [Source X] format naturally in your answer

Example of GOOD response:
"RMI provides bare metal cloud services with high-performance servers featuring 128 vCPUs and 1024 GB RAM. These are ideal for compute-intensive workloads and offer cost savings through reserved instances."

Example of BAD response:
"Looking at the provided documents, I can see that the sources mention RMI provides bare metal services. Based on source 1 and 2..."

Remember: Answer naturally as if you're a knowledgeable colleague, not an AI analyzing documents."""

def format_context(chunks: list) -> str:
    """Format retrieved chunks into context string"""
    context_parts = []
    
    for i, chunk in enumerate(chunks, 1):
        citation = chunk.get('citation', {})
        title = citation.get('title', 'Unknown')
        section = chunk.get('section_path', 'Unknown')
        text = chunk.get('text', '')
        
        context_parts.append(
            f"[Source {i}]\n"
            f"Document: {title}\n"
            f"Section: {section}\n"
            f"Content: {text}\n"
        )
    
    return "\n---\n".join(context_parts)

def format_citations(chunks: list) -> list:
    """Extract and format citations from chunks"""
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
