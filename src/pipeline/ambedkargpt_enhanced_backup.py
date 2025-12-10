"""
Enhanced AmbedkarGPT with better search capabilities
"""

from typing import List, Dict, Any
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
import re
import json
import pickle

class EnhancedAmbedkarGPT:
    """Enhanced version with improved search capabilities"""
    
    def __init__(self):
        self.graph = None
        self.chunks = []
        self.communities = {}
        self.embedding_model = None
        self.llm = None
        print("="*60)
        print("INITIALIZING ENHANCED AMBEDKARGPT")
        print("="*60)
        self.initialize()
    
    def initialize(self):
        """Initialize the system with all components"""
        print("Loading resources...")
        self._load_data()
        self._initialize_models()
        print("✓ System initialized")
        print(f"  • Chunks: {len(self.chunks)}")
        print(f"  • Entities: {self.graph.number_of_nodes() if self.graph else 0}")
        print(f"  • Communities: {len(self.communities)}")
        print("="*60)
    
    def _load_data(self):
        """Load all processed data"""
        try:
            # Load chunks
            with open('../../data/processed/chunks.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.chunks = data.get('chunks', [])
            
            # Load graph
            with open('../../data/processed/graph.gml', 'r', encoding='utf-8') as f:
                self.graph = nx.read_gml(f)
            
            # Load communities
            with open('../../data/processed/communities.pkl', 'rb') as f:
                self.communities = pickle.load(f)
                
            print("✓ Data loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            raise
    
    def _initialize_models(self):
        """Initialize embedding model and LLM"""
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize LLM
        try:
            from langchain.llms import Ollama
            self.llm = Ollama(model="mistral", temperature=0.1)
            print("✓ LLM initialized (Ollama Mistral)")
        except Exception as e:
            print(f"⚠ LLM initialization warning: {e}")
            self.llm = None
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _keyword_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Keyword-based search as fallback"""
        query_words = set(re.findall(r'\w+', query.lower()))
        results = []
        
        # Special handling for known phrases
        special_phrases = {
            'jat pat todak mandal': 0.5,
            'jat-pat-todak': 0.4,
            'reject speech': 0.3
        }
        
        for i, chunk in enumerate(self.chunks):
            text = chunk.get('text', '').lower()
            score = 0.0
            
            # Check for special phrases
            for phrase, boost in special_phrases.items():
                if phrase in text:
                    score += boost
            
            # Calculate word overlap
            chunk_words = set(re.findall(r'\w+', text))
            intersection = len(query_words.intersection(chunk_words))
            union = len(query_words.union(chunk_words))
            
            if union > 0:
                jaccard_score = intersection / union
                score += jaccard_score
            
            if score > 0.1:  # Low threshold
                results.append({
                    'id': i,
                    'text': chunk.get('text', ''),
                    'score': score,
                    'method': 'keyword'
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def local_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Enhanced local search with multiple strategies"""
        print(f"\n[Local Search] Query: '{query}'")
        
        all_results = []
        
        # Strategy 1: Direct semantic search on chunks
        print("  Strategy 1: Semantic search...")
        query_embedding = self.embedding_model.encode([query])[0]
        
        for i, chunk in enumerate(self.chunks):
            text = chunk.get('text', '')
            if len(text) > 50:  # Skip very short chunks
                chunk_embedding = self.embedding_model.encode([text])[0]
                similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                if similarity > 0.15:  # Lower threshold
                    all_results.append({
                        'chunk_id': i,
                        'text': text[:500] + '...' if len(text) > 500 else text,
                        'score': similarity,
                        'method': 'semantic'
                    })
        
        # Strategy 2: Keyword search (especially for proper nouns)
        print("  Strategy 2: Keyword search...")
        keyword_results = self._keyword_search(query, top_k=top_k*2)
        for result in keyword_results:
            all_results.append({
                'chunk_id': result['id'],
                'text': result['text'][:500] + '...' if len(result['text']) > 500 else result['text'],
                'score': result['score'],
                'method': result['method']
            })
        
        # Deduplicate and sort
        unique_results = {}
        for result in all_results:
            chunk_id = result['chunk_id']
            if chunk_id not in unique_results or result['score'] > unique_results[chunk_id]['score']:
                unique_results[chunk_id] = result
        
        final_results = list(unique_results.values())
        final_results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"  ✓ Found {len(final_results)} local chunks")
        return final_results[:top_k]
    
    def global_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Global search - simplified"""
        return self.local_search(query, top_k=top_k)
    
    def combined_search(self, query: str) -> Dict[str, Any]:
        """Combine search results"""
        local_results = self.local_search(query, top_k=5)
        global_results = self.global_search(query, top_k=3)
        
        # Combine and deduplicate
        all_chunks = {}
        for result in local_results + global_results:
            chunk_id = result['chunk_id']
            if chunk_id not in all_chunks or result['score'] > all_chunks[chunk_id]['score']:
                all_chunks[chunk_id] = result
        
        combined_results = list(all_chunks.values())
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        
        search_metrics = {
            'local_chunks_found': len(local_results),
            'global_chunks_found': len(global_results),
            'total_unique_chunks': len(combined_results),
            'query': query
        }
        
        return {
            'combined_results': combined_results[:8],  # Top 8 chunks
            'search_metrics': search_metrics
        }
    
    def generate_answer(self, question: str, context_chunks: List[Dict]) -> str:
        """Generate answer using LLM with context"""
        if not self.llm or not context_chunks:
            return "System not fully initialized or no context found."
        
        # Prepare context
        context_text = "\n\n".join([
            f"[Context {i+1}]: {chunk['text']}"
            for i, chunk in enumerate(context_chunks[:3])  # Use top 3 chunks
        ])
        
        # Create prompt
        prompt = f"""Based on Dr. B.R. Ambedkar's writings, answer the following question.

QUESTION: {question}

CONTEXT FROM AMBEDKAR'S WRITINGS:
{context_text}

INSTRUCTIONS:
1. Answer based ONLY on the provided context
2. Be specific and accurate
3. Cite relevant parts of the context

ANSWER:"""
        
        try:
            response = self.llm(prompt)
            return response.strip()
        except Exception as e:
            return f"Error generating answer: {str(e)}"

# For backward compatibility
AmbedkarGPT = EnhancedAmbedkarGPT