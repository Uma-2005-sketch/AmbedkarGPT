"""
SEMRAG: Semantic RAG with Knowledge Graph Integration
Combines vector retrieval with graph-based reasoning.
"""

import json
from typing import List, Dict, Tuple
from graph_db import MockGraphDB

class SEMRAG:
    """Hybrid retrieval combining vectors and knowledge graph"""
    
    def __init__(self, vector_store=None, graph_db=None):
        self.vector_store = vector_store
        self.graph_db = graph_db or MockGraphDB()
        print("✅ SEMRAG initialized: Vector + Graph retrieval")
    
    def hybrid_retrieve(self, question: str, vector_texts: List[str] = None, top_k: int = 3) -> Dict:
        """Retrieve using both vector similarity and graph semantics"""
        
        # Vector-based retrieval (if available)
        vector_results = []
        if self.vector_store:
            try:
                docs = self.vector_store.similarity_search(question, k=top_k)
                vector_results = [(doc.page_content, doc.metadata.get('source', 'vector')) 
                                 for doc in docs]
            except:
                vector_results = []
        
        # Graph-based retrieval
        graph_results = self.graph_db.semantic_search(question, top_k=top_k)
        
        # Text-based fallback
        text_results = []
        if vector_texts and not vector_results:
            question_lower = question.lower()
            for text in vector_texts:
                if any(word in text.lower() for word in question_lower.split()):
                    text_results.append((text, "corpus"))
            text_results = text_results[:top_k]
        
        # Combine and rank
        all_results = self._combine_results(
            vector_results, graph_results, text_results, question
        )
        
        return {
            "vector_results": vector_results,
            "graph_results": graph_results,
            "text_results": text_results,
            "combined_results": all_results[:top_k]
        }
    
    def _combine_results(self, vector_results: List, graph_results: List, 
                         text_results: List, question: str) -> List[Tuple[str, str, float]]:
        """Combine and rank results from different sources"""
        combined = []
        
        # Add vector results with high weight
        for text, source in vector_results:
            score = 0.9  # High confidence for vector similarity
            combined.append((text, f"vector:{source}", score))
        
        # Add graph results
        for result in graph_results:
            if result['type'] == 'relation':
                text = f"Relation: {result['content']}. Context: {result.get('context', '')}"
            else:
                text = f"Entity: {result['content']} ({result['entity_type']})"
            
            score = result['score'] * 0.7  # Moderate weight for graph
            combined.append((text, f"graph:{result['type']}", score))
        
        # Add text results with lower weight
        for text, source in text_results:
            score = 0.5  # Lower confidence for keyword matching
            combined.append((text, f"text:{source}", score))
        
        # Deduplicate and sort
        unique_results = []
        seen_texts = set()
        for text, source, score in combined:
            if text[:100] not in seen_texts:  # Simple dedup by first 100 chars
                seen_texts.add(text[:100])
                unique_results.append((text, source, score))
        
        # Sort by score
        unique_results.sort(key=lambda x: x[2], reverse=True)
        return unique_results
    
    def get_graph_context(self, question: str) -> str:
        """Get contextual information from knowledge graph"""
        context_parts = []
        
        # Extract key entities from question
        question_lower = question.lower()
        potential_entities = ["ambedkar", "caste", "democracy", "equality", "rights", 
                              "untouchable", "buddha", "constitution"]
        
        for entity in potential_entities:
            if entity in question_lower:
                # Query graph for this entity
                entities = self.graph_db.query_entities(entity)
                relations = self.graph_db.query_relations(entity)
                
                if entities:
                    context_parts.append(f"Knowledge about {entity}:")
                    for e in entities[:2]:
                        context_parts.append(f"  - {e['label']} is a {e['type']}")
                
                if relations:
                    context_parts.append(f"Relations involving {entity}:")
                    for r in relations[:2]:
                        context_parts.append(f"  - {r['source']} {r['relation']} {r['target']}")
        
        if context_parts:
            return "\n".join(context_parts)
        return "No relevant graph context found."
    
    def generate_enhanced_prompt(self, question: str, retrieved_context: str) -> str:
        """Generate enhanced prompt with graph context"""
        graph_context = self.get_graph_context(question)
        
        prompt = f"""You are an expert on Dr. B.R. Ambedkar's philosophy and Indian social reform.

KNOWLEDGE GRAPH CONTEXT (structured knowledge):
{graph_context}

RETRIEVED DOCUMENT CONTEXT (from speeches):
{retrieved_context}

QUESTION: {question}

INSTRUCTIONS:
1. First use the Knowledge Graph Context to understand relationships and entities
2. Then use the Retrieved Document Context for specific quotes and details
3. Synthesize a comprehensive answer that combines both
4. If information conflicts, prioritize the document context
5. If you cannot answer based on either context, say so clearly

ANSWER: """
        
        return prompt

def test_semrag():
    """Test SEMRAG functionality"""
    print("="*60)
    print("SEMRAG TEST: Hybrid Vector + Graph Retrieval")
    print("="*60)
    
    # Initialize SEMRAG (without vector store for now)
    semrag = SEMRAG()
    
    test_questions = [
        "What is caste system?",
        "What did Ambedkar say about democracy?",
        "Who was Buddha?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        
        # Get graph context
        graph_context = semrag.get_graph_context(question)
        print(f"   Graph Context: {graph_context[:100]}...")
        
        # Simulate retrieval
        mock_retrieved = "Ambedkar said that caste system is a social evil that must be destroyed through education and political empowerment."
        
        # Generate enhanced prompt
        prompt = semrag.generate_enhanced_prompt(question, mock_retrieved)
        print(f"   Prompt length: {len(prompt)} chars")
        print(f"   Prompt preview: {prompt[:150]}...")
    
    print("\n✅ SEMRAG test complete!")

if __name__ == "__main__":
    test_semrag()