"""
Answer generator for SEMRAG system
Combines retrieval results with LLM generation
"""
import time
from typing import Dict, List, Any, Optional
from .llm_client import LLMClient
from .prompt_templates import PromptTemplates

class AnswerGenerator:
    """
    Generates answers by combining retrieval results with LLM
    """
    
    def __init__(self, model="mistral", temperature=0.1):
        """
        Args:
            model: LLM model name
            temperature: Sampling temperature
        """
        self.llm_client = LLMClient(model=model, temperature=temperature)
        self.prompt_templates = PromptTemplates()
        self.conversation_history = []
        
    def generate_from_local_rag(self, query: str, local_results: Dict, 
                               context: str = "") -> Dict:
        """
        Generate answer from local RAG results
        
        Args:
            query: User query
            local_results: Results from local search (Equation 4)
            context: Additional context
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        # Create prompt
        prompt = self.prompt_templates.local_rag_prompt(
            query=query,
            retrieved_items=local_results.get('retrieved_items', []),
            context=context
        )
        
        # Generate answer
        system_prompt = self.prompt_templates.SYSTEM_PROMPTS["ambedkar_expert"]
        answer = self.llm_client.generate(prompt, system_prompt)
        
        elapsed = time.time() - start_time
        
        # Build response
        response = {
            'query': query,
            'answer': answer,
            'retrieval_method': 'local_graph',
            'retrieval_parameters': {
                'tau_e': local_results.get('tau_e', 0.6),
                'tau_d': local_results.get('tau_d', 0.5),
                'top_k': local_results.get('top_k', 5)
            },
            'retrieval_stats': {
                'entities_found': len(set(
                    item.get('entity_label', '') 
                    for item in local_results.get('retrieved_items', [])
                )),
                'chunks_retrieved': len(local_results.get('retrieved_items', [])),
                'top_entity': next(iter([
                    item.get('entity_label', '') 
                    for item in local_results.get('retrieved_items', [])[:1]
                ]), 'None')
            },
            'generation_time': elapsed,
            'model': self.llm_client.model
        }
        
        # Add to conversation history
        self.conversation_history.append({
            'query': query,
            'answer': answer[:200] + '...' if len(answer) > 200 else answer,
            'method': 'local_graph',
            'timestamp': time.time()
        })
        
        return response
    
    def generate_from_global_rag(self, query: str, global_results: Dict) -> Dict:
        """
        Generate answer from global RAG results
        
        Args:
            query: User query
            global_results: Results from global search (Equation 5)
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        # Extract community summaries
        community_summaries = []
        for point in global_results.get('retrieved_points', [])[:3]:
            summary = point.get('community_summary', '')
            if summary and summary not in community_summaries:
                community_summaries.append({'summary': summary})
        
        # Create prompt
        prompt = self.prompt_templates.global_rag_prompt(
            query=query,
            community_summaries=community_summaries,
            retrieved_points=global_results.get('retrieved_points', [])
        )
        
        # Generate answer
        system_prompt = self.prompt_templates.SYSTEM_PROMPTS["ambedkar_expert"]
        answer = self.llm_client.generate(prompt, system_prompt)
        
        elapsed = time.time() - start_time
        
        # Build response
        response = {
            'query': query,
            'answer': answer,
            'retrieval_method': 'global_graph',
            'retrieval_parameters': {
                'top_k_communities': global_results.get('top_k_communities', 3),
                'top_k_points': global_results.get('top_k_points', 5)
            },
            'retrieval_stats': {
                'communities_considered': len(global_results.get('top_communities', [])),
                'points_retrieved': len(global_results.get('retrieved_points', [])),
                'top_community': next(iter([
                    f"Community {comm.get('community_id', '')}"
                    for comm in global_results.get('top_communities', [])[:1]
                ]), 'None')
            },
            'generation_time': elapsed,
            'model': self.llm_client.model
        }
        
        # Add to conversation history
        self.conversation_history.append({
            'query': query,
            'answer': answer[:200] + '...' if len(answer) > 200 else answer,
            'method': 'global_graph',
            'timestamp': time.time()
        })
        
        return response
    
    def generate_hybrid_answer(self, query: str, local_results: Dict, 
                              global_results: Dict) -> Dict:
        """
        Generate answer combining both local and global RAG results
        
        Args:
            query: User query
            local_results: Results from local search
            global_results: Results from global search
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        # Create hybrid prompt
        prompt = self.prompt_templates.hybrid_rag_prompt(
            query=query,
            local_results=local_results,
            global_results=global_results
        )
        
        # Generate answer
        system_prompt = self.prompt_templates.SYSTEM_PROMPTS["ambedkar_expert"]
        answer = self.llm_client.generate(prompt, system_prompt)
        
        elapsed = time.time() - start_time
        
        # Build response
        response = {
            'query': query,
            'answer': answer,
            'retrieval_method': 'hybrid',
            'local_stats': {
                'entities_found': len(set(
                    item.get('entity_label', '') 
                    for item in local_results.get('retrieved_items', [])
                )),
                'chunks_retrieved': len(local_results.get('retrieved_items', []))
            },
            'global_stats': {
                'communities_considered': len(global_results.get('top_communities', [])),
                'points_retrieved': len(global_results.get('retrieved_points', []))
            },
            'generation_time': elapsed,
            'model': self.llm_client.model
        }
        
        # Add to conversation history
        self.conversation_history.append({
            'query': query,
            'answer': answer[:200] + '...' if len(answer) > 200 else answer,
            'method': 'hybrid',
            'timestamp': time.time()
        })
        
        return response
    
    def get_conversation_history(self, limit: int = 5) -> List[Dict]:
        """
        Get recent conversation history
        
        Args:
            limit: Number of recent exchanges to return
            
        Returns:
            List of conversation exchanges
        """
        return self.conversation_history[-limit:]
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()

# Test function
def test_answer_generator():
    """Test answer generator"""
    print("ANSWER GENERATOR TEST")
    print("=" * 60)
    
    # Create generator
    generator = AnswerGenerator(model="mistral", temperature=0.1)
    
    # Mock retrieval results (simplified for testing)
    local_results = {
        'retrieved_items': [
            {
                'entity_label': 'Ambedkar',
                'entity_type': 'person',
                'chunk_text': 'Dr. B.R. Ambedkar criticized the caste system as a hierarchical structure.',
                'combined_score': 0.85
            },
            {
                'entity_label': 'caste',
                'entity_type': 'concept',
                'chunk_text': 'The caste system divides society into watertight compartments.',
                'combined_score': 0.78
            }
        ],
        'tau_e': 0.6,
        'tau_d': 0.5,
        'top_k': 5
    }
    
    global_results = {
        'retrieved_points': [
            {
                'community_id': 0,
                'community_summary': 'Community about social reform and caste annihilation',
                'chunk_text': 'Ambedkar advocated for the annihilation of caste through rejection of religious scriptures.',
                'combined_score': 0.82
            }
        ],
        'top_communities': [
            {'community_id': 0, 'score': 0.82}
        ],
        'top_k_communities': 3,
        'top_k_points': 5
    }
    
    # Test 1: Local RAG answer
    print("\n1. LOCAL RAG ANSWER GENERATION:")
    print("-" * 60)
    
    local_response = generator.generate_from_local_rag(
        "What is caste system according to Ambedkar?",
        local_results
    )
    
    print(f"Query: {local_response['query']}")
    print(f"\nRetrieval Method: {local_response['retrieval_method']}")
    print(f"Entities found: {local_response['retrieval_stats']['entities_found']}")
    print(f"Chunks retrieved: {local_response['retrieval_stats']['chunks_retrieved']}")
    print(f"Generation time: {local_response['generation_time']:.2f}s")
    print(f"\nAnswer (first 300 chars):")
    print("-" * 30)
    print(local_response['answer'][:300] + "..." if len(local_response['answer']) > 300 else local_response['answer'])
    
    # Test 2: Global RAG answer
    print("\n\n2. GLOBAL RAG ANSWER GENERATION:")
    print("-" * 60)
    
    global_response = generator.generate_from_global_rag(
        "How did Ambedkar propose to reform society?",
        global_results
    )
    
    print(f"Query: {global_response['query']}")
    print(f"\nRetrieval Method: {global_response['retrieval_method']}")
    print(f"Communities considered: {global_response['retrieval_stats']['communities_considered']}")
    print(f"Points retrieved: {global_response['retrieval_stats']['points_retrieved']}")
    print(f"Generation time: {global_response['generation_time']:.2f}s")
    print(f"\nAnswer (first 300 chars):")
    print("-" * 30)
    print(global_response['answer'][:300] + "..." if len(global_response['answer']) > 300 else global_response['answer'])
    
    # Test 3: Hybrid answer
    print("\n\n3. HYBRID ANSWER GENERATION:")
    print("-" * 60)
    
    hybrid_response = generator.generate_hybrid_answer(
        "What was Ambedkar's approach to caste and social reform?",
        local_results,
        global_results
    )
    
    print(f"Query: {hybrid_response['query']}")
    print(f"\nRetrieval Method: {hybrid_response['retrieval_method']}")
    print(f"Local entities: {hybrid_response['local_stats']['entities_found']}")
    print(f"Global communities: {hybrid_response['global_stats']['communities_considered']}")
    print(f"Generation time: {hybrid_response['generation_time']:.2f}s")
    print(f"\nAnswer (first 300 chars):")
    print("-" * 30)
    print(hybrid_response['answer'][:300] + "..." if len(hybrid_response['answer']) > 300 else hybrid_response['answer'])
    
    # Show conversation history
    print("\n\n4. CONVERSATION HISTORY:")
    print("-" * 60)
    
    history = generator.get_conversation_history()
    for i, exchange in enumerate(history, 1):
        print(f"\nExchange {i}:")
        print(f"  Query: {exchange['query'][:50]}...")
        print(f"  Method: {exchange['method']}")
        print(f"  Answer preview: {exchange['answer'][:80]}...")
    
    print("\n✅ Answer generator testing complete!")
    return generator

if __name__ == "__main__":
    test_answer_generator()