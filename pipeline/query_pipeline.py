from typing import Dict, List
import re
from security.redaction import compile_patterns, scrub

class QueryPipeline:
    def __init__(self, db, hybrid_search, reranker, llm_client, config):
        self.db = db
        self.hybrid_search = hybrid_search
        self.reranker = reranker
        self.llm_client = llm_client
        self.config = config
        self.redaction_patterns = compile_patterns()
    
    def query(self, question: str, history: List[Dict] = None) -> Dict:
        if history is None:
            history = []
        
        try:
            # Summarize long history
            if len(history) > 10:
                history = self._summarize_history(history)
            
            # Rewrite query for follow-ups
            expanded_query = self._rewrite_query(question, history)
            
            # Hybrid retrieval
            results = self.hybrid_search.search(expanded_query)
            
            # Check if results is empty (handle list properly)
            if not results or len(results) == 0:
                return self._handle_no_results(question, history)
            
            # Fetch chunks
            top_k = min(40, len(results))
            chunks = self._fetch_chunks([cid for cid, _ in results[:top_k]])
            
            # Check if chunks is empty
            if not chunks or len(chunks) == 0:
                return self._handle_no_results(question, history)
            
            # Pre-scrub chunks
            for chunk in chunks:
                chunk['text'] = scrub(chunk['text'], self.redaction_patterns)
            
            # Rerank
            if self.reranker:
                reranked = self.reranker.rerank(expanded_query, chunks, self.config.rerank.top_n)
                
                # Check if reranked is empty
                if not reranked or len(reranked) == 0:
                    return self._handle_no_results(question, history)
                
                top_score = reranked[0][1]
                final_chunks = [chunk for chunk, _ in reranked]
                
                # Answerability check
                if not self._can_answer(question, final_chunks, top_score):
                    # If no good docs found, use GENERAL mode to allow LLM to respond naturally
                    return self._generate_answer_with_mode(question, [], history, "GENERAL")
                
                return self._generate_answer_with_mode(question, final_chunks, history, "DOC-GROUNDED")
            else:
                final_chunks = chunks[:self.config.rerank.top_n]
                return self._generate_answer_with_mode(question, final_chunks, history, "DOC-GROUNDED")
                
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "answer": "An error occurred while processing your question. Please try again.",
                "citations": [],
                "confidence": "low",
                "chunks_used": 0,
                "mode": "GENERAL"
            }
    
    def _rewrite_query(self, question: str, history: List[Dict]) -> str:
        if not history or not self._is_followup_question(question):
            return question
        
        prompt = "Rewrite the user's latest question into a standalone query.\n\nPrevious turns:\n"
        last_n = history[-6:]
        for m in last_n:
            role = m["role"]
            content = m["content"][:200]
            prompt += f"{role.title()}: {content}\n"
        prompt += f"\nUser: {question}\nStandalone query:"
        
        try:
            resp = self.llm_client.generate(
                system="You rewrite queries into standalone form.",
                user=prompt,
                context="",
                history=""
            )
            rewritten = (resp.get("response") or "").strip()
            return rewritten if rewritten else question
        except Exception as e:
            print(f"Warning: Query rewrite failed ({e}), using original question")
            return question
    
    def _summarize_history(self, history: List[Dict]) -> List[Dict]:
        if len(history) < 10:
            return history
        
        sample = history[-8:]
        text = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in sample)
        prompt = "Summarize this dialogue into bullet points with key entities and current goal."
        
        try:
            resp = self.llm_client.generate(
                system="You summarize crisply.",
                user=prompt,
                context=text,
                history=""
            )
            memo = resp.get("response", "").strip()
            if memo:
                return history[:-8] + [{"role": "assistant", "content": f"[MEMO]\n{memo}"}]
            else:
                print("Warning: History summarization returned empty, keeping original")
                return history
        except Exception as e:
            print(f"Warning: History summarization failed ({e}), keeping original")
            return history
    
    def _can_answer(self, question: str, chunks: List[Dict], top_score: float) -> bool:
        """
        Determine if the retrieved chunks can answer the question
        Very lenient - prefer using docs when available
        """
        try:
            # Basic threshold check
            if top_score < self.config.abstain.min_rerank_score:
                return False
            if len(chunks) < self.config.abstain.min_chunks:
                return False
            
            # If we have ANY decent chunks (>0.3), use them
            # The reranker already filtered for relevance
            return True
            
        except Exception as e:
            print(f"Warning: Answerability check failed ({e}), defaulting to True")
            return True
    
    def _is_followup_question(self, question: str) -> bool:
        followup_indicators = [
            r'\b(it|this|that|these|those|its|their|them)\b',
            r'\b(last|previous|earlier|above|before)\b',
            r'^(what|how|why|when|where|who)\s+(about|was|is|are)\b',
            r'^(tell me|explain|describe)\s+more\b',
            r'^(and|also|additionally)\b',
        ]
        question_lower = question.lower()
        return any(re.search(pattern, question_lower, re.IGNORECASE) for pattern in followup_indicators)
    
    def _generate_answer_with_mode(self, question: str, chunks: List[Dict], 
                                   history: List[Dict], mode: str) -> Dict:
        from llm.prompts import SYSTEM_PROMPT, build_user_prompt, format_context, format_citations
        
        context = format_context(chunks) if chunks else ""
        history_context = self._format_history_for_llm(history)
        
        user_prompt = build_user_prompt(question, context, mode, history_context)
        
        try:
            response = self.llm_client.generate(
                system=SYSTEM_PROMPT,
                user=user_prompt,
                context="",
                history=""
            )
            
            if "error" in response:
                return {
                    "answer": "Error generating response. Please try again.",
                    "citations": [],
                    "confidence": "low",
                    "chunks_used": 0,
                    "mode": "GENERAL"
                }
            
            answer_text = response.get("response", "").strip()
            answer_text = scrub(answer_text, self.redaction_patterns)
            
            confidence = self._assess_confidence(answer_text, chunks, mode)
            
            return {
                "answer": answer_text,
                "citations": format_citations(chunks) if chunks else [],
                "confidence": confidence,
                "chunks_used": len(chunks),
                "mode": mode
            }
        except Exception as e:
            print(f"Error generating answer: {e}")
            import traceback
            traceback.print_exc()
            return {
                "answer": "An error occurred while generating the response. Please try again.",
                "citations": [],
                "confidence": "low",
                "chunks_used": 0,
                "mode": "GENERAL"
            }
    
    def _format_history_for_llm(self, history: List[Dict], last_n: int = 3) -> str:
        if not history:
            return ""
        
        recent_history = history[-last_n*2:]
        if not recent_history:
            return ""
        
        formatted = []
        for msg in recent_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                content_short = content[:200] + "..." if len(content) > 200 else content
                formatted.append(f"Assistant: {content_short}")
        
        return "\n".join(formatted)
    
    def _handle_no_results(self, question: str, history: List[Dict]) -> Dict:
        # Always use GENERAL mode to let LLM respond naturally
        return self._generate_answer_with_mode(question, [], history, "GENERAL")
    
    def _assess_confidence(self, answer: str, chunks: List[Dict], mode: str) -> str:
        if mode == "GENERAL":
            return "general"
        if len(chunks) >= 5:
            return "high"
        elif len(chunks) >= 3:
            return "medium"
        else:
            return "low"
    
    def _fetch_chunks(self, chunk_ids: List[int]) -> List[Dict]:
        chunks = []
        for chunk_id in chunk_ids:
            try:
                chunk = self.db.get_chunk(chunk_id)
                if chunk:
                    citation = self.db.get_citation(chunk_id)
                    chunk['citation'] = citation
                    chunks.append(chunk)
            except Exception as e:
                print(f"Warning: Failed to fetch chunk {chunk_id}: {e}")
                continue
        return chunks
