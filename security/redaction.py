import re
from typing import List

SENSITIVE_PATTERNS = [
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
    r'\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{4}\b',
    r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    r'\b([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',
    r'\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b',
    r'AKIA[0-9A-Z]{16}',
    r'ASIA[0-9A-Z]{16}',
    r'(?i)aws_secret_access_key\s*=\s*[A-Za-z0-9/+=]{40}',
    r'xox[baprs]-[A-Za-z0-9-]{10,48}',
    r'https://hooks\.slack\.com/services/[A-Za-z0-9/_-]+',
    r'(?:api[_-]?key|secret|password|token|bearer|jwt)[^\n]{0,50}',
    r'-----BEGIN (?:RSA|DSA|EC|OPENSSH) PRIVATE KEY-----[\s\S]+?-----END .* PRIVATE KEY-----',
    r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
]

def compile_patterns() -> List[re.Pattern]:
    """Compile all sensitive data regex patterns"""
    return [re.compile(p) for p in SENSITIVE_PATTERNS]

def scrub(text: str, patterns: List[re.Pattern] = None) -> str:
    """
    Scrub sensitive information from text
    
    Args:
        text: Text to scrub
        patterns: Optional pre-compiled patterns. If None, will compile fresh.
    
    Returns:
        Scrubbed text with [REDACTED] replacing sensitive data
    """
    if patterns is None:
        patterns = compile_patterns()
    
    for pattern in patterns:
        text = pattern.sub('[REDACTED]', text)
    
    return text
