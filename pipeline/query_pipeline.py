from typing import Dict, List, Optional
import re

class QueryPipeline:
    """Complete RAG query pipeline with simple conversation memory"""
    
    def __init__(self, db, hybrid_search, reranker, llm_client, config):
        self.db = db
        self.hybrid_search = hybrid_search
        self.reranker = reranker
        self.llm_client = llm_client
        self.config = config
        self.redaction_patterns = self._compile_redaction_patterns()
    
    def query(self, question: str, history: List[Dict] = None) -> Dict:
        """
        Execute complete RAG pipeline with conversation context
        
        Args:
            question: Current user question
            history: List of previous turns in format [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        """
        if history is None:
            history = []
        
        # Expand query with conversation context if this is a follow-up
        expanded_query = self._expand_query_with_history(question, history)
        
        # Step 1: Hybrid retrieval
        results = self.hybrid_search.search(expanded_query)
        
        if not results:
            return self._request_clarification(question, history)
        
        # Step 2: Fetch chunk details
        top_k = min(50, len(results))
        chunks = self._fetch_chunks([cid for cid, _ in results[:top_k]])
        
        if not chunks:
            return self._request_clarification(question, history)
        
        # Step 3: Rerank if enabled
        if self.reranker:
            reranked = self.reranker.rerank(
                expanded_query, 
                chunks, 
                self.config.rerank.top_n
            )
            
            if not reranked:
                return self._request_clarification(question, history)
            
            top_score = reranked[0][1]
            
            # Lower threshold for follow-up questions
            threshold = self.config.abstain.min_rerank_score
            if history and self._is_followup_question(question):
                threshold *= 0.7  # Be more lenient
            
            if top_score < threshold:
                return self._request_clarification(question, history)
            
            if len(reranked) < self.config.abstain.min_chunks:
                return self._request_clarification(question, history)
            
            final_chunks = [chunk for chunk, _ in reranked]
        else:
            final_chunks = chunks[:self.config.rerank.top_n]
        
        # Step 4: Generate answer with conversation history
        return self._generate_answer(question, final_chunks, history)
    
    def _expand_query_with_history(self, question: str, history: List[Dict]) -> str:
        """Expand query using conversation history"""
        if not history:
            return question
        
        # Get last user question
        last_user_msg = None
        for msg in reversed(history):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break
        
        if not last_user_msg:
            return question
        
        # If current question is a follow-up, combine with previous context
        if self._is_followup_question(question):
            return f"{last_user_msg} {question}"
        
        return question
    
    def _is_followup_question(self, question: str) -> bool:
        """Check if question is a follow-up or refers to previous context"""
        followup_indicators = [
            # Pronouns and references
            r'\b(it|this|that|these|those|its|their|them)\b',
            r'\b(last|previous|earlier|above|before)\b',
            
            # Question patterns that need context
            r'^(what|how|why|when|where|who)\s+(about|was|is|are)\b',
            r'^(tell me|explain|describe)\s+more\b',
            r'^(and|also|additionally)\b',
            
            # Meta questions about conversation
            r'\b(my|our)\s+(question|query|last)\b',
            r'what (did|was) (i|we)',
        ]
        
        question_lower = question.lower()
        return any(re.search(pattern, question_lower, re.IGNORECASE) for pattern in followup_indicators)
    
    def _request_clarification(self, question: str, history: List[Dict]) -> Dict:
        """Request clarification instead of immediately abstaining"""
        
        # Check if question is too vague
        vague_patterns = [
            r'^\s*(what|how|tell|explain|describe)\s*\??\s*$',
            r'^\s*(price|pricing|cost|费用)\s*\??\s*$',
            r'^\s*(it|this|that)\s*\??\s*$',
        ]
        
        question_stripped = question.strip().lower()
        is_vague = any(re.match(pattern, question_stripped, re.IGNORECASE) for pattern in vague_patterns)
        
        if is_vague:
            # Get context hint from history
            hint = ""
            if history:
                for msg in reversed(history):
                    if msg.get("role") == "user":
                        last_q = msg.get("content", "")[:50]
                        hint = f" (related to your previous question about: '{last_q}...')"
                        break
            
            return {
                "answer": f"I need more details to help you. Could you please be more specific?{hint}",
                "citations": [],
                "confidence": "clarification_needed",
                "chunks_used": 0
            }
        
        return self._abstain_response("No relevant information found in documentation.")
    
    def _fetch_chunks(self, chunk_ids: List[int]) -> List[Dict]:
        """Fetch chunk details with citations"""
        chunks = []
        for chunk_id in chunk_ids:
            chunk = self.db.get_chunk(chunk_id)
            if chunk:
                citation = self.db.get_citation(chunk_id)
                chunk['citation'] = citation
                chunks.append(chunk)
        return chunks
    
    def _generate_answer(self, question: str, chunks: List[Dict], 
                        history: List[Dict]) -> Dict:
        """Generate answer using LLM with conversation context"""
        from llm.prompts import SYSTEM_PROMPT, format_context, format_citations
        
        context = format_context(chunks)
        
        # Format conversation history for LLM
        history_context = self._format_history_for_llm(history)
        
        response = self.llm_client.generate(
            system=SYSTEM_PROMPT,
            user=question,
            context=context,
            history=history_context
        )
        
        if "error" in response:
            return {
                "answer": "Error generating response. Please try again.",
                "citations": [],
                "confidence": "low",
                "error": response["error"]
            }
        
        answer_text = response.get("response", "").strip()
        
        # Apply redaction
        answer_text = self._redact_sensitive_info(answer_text)
        
        confidence = self._assess_confidence(answer_text, chunks)
        
        return {
            "answer": answer_text,
            "citations": format_citations(chunks),
            "confidence": confidence,
            "chunks_used": len(chunks)
        }
    
    def _format_history_for_llm(self, history: List[Dict], last_n: int = 3) -> str:
        """Format conversation history for LLM prompt"""
        if not history:
            return ""
        
        # Get last N turns (pair of user + assistant)
        recent_history = history[-last_n*2:] if len(history) > last_n*2 else history
        
        if not recent_history:
            return ""
        
        formatted = ["Previous conversation:"]
        for msg in recent_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                formatted.append(f"\nUser: {content}")
            elif role == "assistant":
                # Truncate long assistant responses
                content_short = content[:200] + "..." if len(content) > 200 else content
                formatted.append(f"Assistant: {content_short}")
        
        return "\n".join(formatted)
    
    def _abstain_response(self, reason: str) -> Dict:
        """Return abstain response"""
        return {
            "answer": "This information is not available in our documentation.",
            "citations": [],
            "confidence": "abstain",
            "reason": reason
        }
    
    def _assess_confidence(self, answer: str, chunks: List[Dict]) -> str:
        """Assess answer confidence"""
        if "not available" in answer.lower() or "cannot answer" in answer.lower():
            return "low"
        
        if len(chunks) >= 5:
            return "high"
        elif len(chunks) >= 3:
            return "medium"
        else:
            return "low"
    
    def _compile_redaction_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for sensitive information"""
        patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            r'\b\(\d{3}\)\s?\d{3}[-.]?\d{4}\b',
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            r'\b[A-Za-z0-9_-]{32,}\b',
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            r'\b\d{3}-\d{2}-\d{4}\b',
        ]
        
        return [re.compile(p) for p in patterns]
    
    def _redact_sensitive_info(self, text: str) -> str:
        """Redact sensitive information from text"""
        for pattern in self.redaction_patterns:
            text = pattern.sub('[REDACTED]', text)
        return text