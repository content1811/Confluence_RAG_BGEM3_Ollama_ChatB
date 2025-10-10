import requests
import json
import re
from typing import Dict, Optional

class LocalLLMClient:
    """Client for Ollama-hosted local LLMs"""
    
    def __init__(self, base_url: str, model: str, temperature: float, max_tokens: int):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    def generate(self, system: str, user: str, context: Optional[str] = None,
                 history: Optional[str] = None) -> Dict:
        """Generate response from local LLM with conversation history"""
        
        prompt = self._compose_prompt(system, user, context, history)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            
            # Clean the response
            if 'response' in result:
                result['response'] = self._clean_response(result['response'])
            
            return result
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def _compose_prompt(self, system: str, user: str, 
                       context: Optional[str], history: Optional[str]) -> str:
        """Compose prompt with system, history, context, and user message"""
        parts = [f"System: {system}"]
        
        if history:
            parts.append(f"\n{history}")
        
        if context:
            parts.append(f"\nContext:\n{context}")
        
        parts.append(f"\nUser: {user}")
        parts.append("\nAssistant:")
        
        return "\n".join(parts)
    
    def _clean_response(self, text: str) -> str:
        """Remove internal reasoning tags and clean response"""
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'\[thinking\].*?\[/thinking\]', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        meta_patterns = [
            r'^(Okay|Alright|So|Well),?\s+(let me|I need to|I should|I\'ll)\s+.+?\.(\s|$)',
            r'^Looking at (the )?(sources?|documents?|context|information provided).+?\.(\s|$)',
            r'^Based on (the )?(sources?|documents?|context|information provided).+?\.(\s|$)',
            r'^From (the )?(sources?|documents?|context|information provided).+?\.(\s|$)',
            r'^According to (the )?(sources?|documents?|context|information provided).+?\.(\s|$)',
        ]
        
        for pattern in meta_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        
        return text