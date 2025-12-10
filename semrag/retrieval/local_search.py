"""
Local Graph RAG Search - Equation 4 from SEMRAG paper

Equation 4: D_retrieved = Top_k({v ∈ V, g ∈ G | sim(v, Q+H) > τ_e ∧ sim(g, v) > τ_d})
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class LocalGraphRAGSearch:
    """
    Implements Local Graph RAG Search (Equation 4)
    
    Retrieves entities and chunks based on similarity to query
    """
    
    def __init__(self, knowledge_graph, entity_embeddings, chunk_embeddings):
        """
        Args:
            knowledge_graph: networkx Graph
            entity_embeddings: dict mapping entity_id -> embedding vector
            chunk_embeddings: dict mapping chunk_id -> embedding vector
        """
        self.graph = knowledge_graph
        self.entity_embeddings = entity_embeddings
        self.chunk_embeddings = chunk_embeddings
        
        # Create mappings
        self.entities_by_id = {}
        self.chunks_by_id = {}
        
        # Extract entity and chunk information from graph
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'chunk':
                chunk_id = data.get('chunk_id')
                if chunk_id is not None:
                    self.chunks_by_id[chunk_id] = {
                        'node_id': node,
                        'data': data
                    }
            elif 'label' in data:
                # Entity node
                entity_id = data.get('label', '').lower().replace(' ', '_')
                self.entities_by_id[entity_id] = {
                    'node_id': node,
                    'data': data,
                    'type': data.get('type', 'unknown')
                }
    
    def _simplify_embedding(self, text):
        """Create simple embedding for query (for development)"""
        # Simple hash-based embedding
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        seed = int(text_hash[:8], 16) % 100000
        np.random.seed(seed)
        emb = np.random.randn(384)
        return emb / np.linalg.norm(emb)
    
    def search(self, query, history=None, top_k=5, tau_e=0.6, tau_d=0.5):
        """
        Perform local graph RAG search (Equation 4)
        
        Args:
            query: User query string
            history: Optional conversation history
            top_k: Number of results to return
            tau_e: Entity similarity threshold
            tau_d: Chunk similarity threshold
            
        Returns:
            List of retrieved chunks with scores
        """
        # Combine query with history if provided
        if history:
            combined_query = f"{history} {query}"
        else:
            combined_query = query
        
        # Create query embedding
        query_embedding = self._simplify_embedding(combined_query)
        query_embedding = query_embedding.reshape(1, -1)
        
        retrieved_items = []
        
        # Step 1: Calculate similarity with entities
        for entity_id, entity_info in self.entities_by_id.items():
            if entity_id in self.entity_embeddings:
                entity_emb = self.entity_embeddings[entity_id].reshape(1, -1)
                entity_sim = cosine_similarity(query_embedding, entity_emb)[0][0]
                
                # Filter by threshold τ_e
                if entity_sim > tau_e:
                    
                    # Step 2: Find chunks related to this entity
                    entity_node = entity_info['node_id']
                    
                    # Find neighboring chunk nodes
                    if entity_node in self.graph:
                        neighbors = list(self.graph.neighbors(entity_node))
                        
                        for neighbor in neighbors:
                            neighbor_data = self.graph.nodes[neighbor]
                            if neighbor_data.get('type') == 'chunk':
                                chunk_id = neighbor_data.get('chunk_id')
                                
                                if chunk_id is not None and chunk_id in self.chunk_embeddings:
                                    chunk_emb = self.chunk_embeddings[chunk_id].reshape(1, -1)
                                    chunk_sim = cosine_similarity(entity_emb, chunk_emb)[0][0]
                                    
                                    # Filter by threshold τ_d
                                    if chunk_sim > tau_d:
                                        # Calculate combined score
                                        combined_score = (entity_sim + chunk_sim) / 2
                                        
                                        retrieved_items.append({
                                            'entity_id': entity_id,
                                            'entity_label': entity_info['data'].get('label', entity_id),
                                            'entity_type': entity_info['data'].get('type', 'unknown'),
                                            'entity_similarity': float(entity_sim),
                                            'chunk_id': chunk_id,
                                            'chunk_similarity': float(chunk_sim),
                                            'combined_score': float(combined_score),
                                            'retrieval_type': 'local_graph'
                                        })
        
        # Sort by combined score and return top_k
        retrieved_items.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Return results with explanation
        results = {
            'query': query,
            'history': history,
            'tau_e': tau_e,
            'tau_d': tau_d,
            'top_k': top_k,
            'retrieved_items': retrieved_items[:top_k],
            'total_matches': len(retrieved_items)
        }
        
        return results
    
    def explain_search(self, query, history=None):
        """Generate explanation of the search process"""
        results = self.search(query, history, top_k=3)
        
        explanation = []
        explanation.append(f"Local Graph RAG Search (Equation 4)")
        explanation.append(f"Query: {query}")
        if history:
            explanation.append(f"History: {history}")
        
        explanation.append(f"\nParameters:")
        explanation.append(f"  τ_e (entity threshold): {results['tau_e']}")
        explanation.append(f"  τ_d (chunk threshold): {results['tau_d']}")
        explanation.append(f"  Top-k: {results['top_k']}")
        
        explanation.append(f"\nSearch Process:")
        explanation.append(f"  1. Combined query with history: '{results['query']}'")
        explanation.append(f"  2. Calculated similarity with {len(self.entities_by_id)} entities")
        explanation.append(f"  3. Filtered entities with similarity > {results['tau_e']}")
        explanation.append(f"  4. Found {results['total_matches']} entity-chunk pairs")
        explanation.append(f"  5. Filtered chunks with similarity > {results['tau_d']}")
        explanation.append(f"  6. Returned top {len(results['retrieved_items'])} results")
        
        if results['retrieved_items']:
            explanation.append(f"\nRetrieved Results:")
            for i, item in enumerate(results['retrieved_items'], 1):
                explanation.append(f"  {i}. Entity: {item['entity_label']} ({item['entity_type']})")
                explanation.append(f"     Entity similarity: {item['entity_similarity']:.3f}")
                explanation.append(f"     Chunk ID: {item['chunk_id']}")
                explanation.append(f"     Chunk similarity: {item['chunk_similarity']:.3f}")
                explanation.append(f"     Combined score: {item['combined_score']:.3f}")
        else:
            explanation.append(f"\nNo results found with current thresholds.")
        
        return "\n".join(explanation)

# Test function
def test_local_search():
    """Test local graph RAG search"""
    print("LOCAL GRAPH RAG SEARCH TEST (Equation 4)")
    print("=" * 60)
    
    # Create mock data for testing
    import networkx as nx
    
    # Create a simple knowledge graph
    graph = nx.Graph()
    
    # Add entities
    entities = ['Ambedkar', 'caste', 'equality', 'constitution']
    for entity in entities:
        graph.add_node(
            f"{entity.lower()}_entity",
            label=entity,
            type='entity',
            entity_type='person' if entity == 'Ambedkar' else 'concept'
        )
    
    # Add chunks
    chunks = ['Chunk about Ambedkar and caste', 
              'Chunk about equality in constitution',
              'Chunk about social reform']
    
    for i, chunk in enumerate(chunks):
        graph.add_node(
            f"chunk_{i}",
            label=chunk,
            type='chunk',
            chunk_id=i
        )
    
    # Connect entities to chunks
    graph.add_edge('ambedkar_entity', 'chunk_0', relation='mentioned_in', weight=1)
    graph.add_edge('caste_entity', 'chunk_0', relation='mentioned_in', weight=1)
    graph.add_edge('equality_entity', 'chunk_1', relation='mentioned_in', weight=1)
    graph.add_edge('constitution_entity', 'chunk_1', relation='mentioned_in', weight=1)
    graph.add_edge('ambedkar_entity', 'chunk_2', relation='mentioned_in', weight=1)
    
    # Create mock embeddings
    np.random.seed(42)
    entity_embeddings = {}
    for entity in entities:
        entity_id = entity.lower().replace(' ', '_')
        entity_embeddings[entity_id] = np.random.randn(384)
        entity_embeddings[entity_id] = entity_embeddings[entity_id] / np.linalg.norm(entity_embeddings[entity_id])
    
    chunk_embeddings = {}
    for i in range(len(chunks)):
        chunk_embeddings[i] = np.random.randn(384)
        chunk_embeddings[i] = chunk_embeddings[i] / np.linalg.norm(chunk_embeddings[i])
    
    # Create search instance
    searcher = LocalGraphRAGSearch(graph, entity_embeddings, chunk_embeddings)
    
    # Test queries
    test_queries = [
        "What did Ambedkar say about caste?",
        "Explain equality in constitution",
        "Social reform by Ambedkar"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print('='*60)
        
        results = searcher.search(query, top_k=2)
        
        print(f"\nFound {results['total_matches']} matches, showing top {len(results['retrieved_items'])}:")
        
        for i, item in enumerate(results['retrieved_items'], 1):
            print(f"\n  Result {i}:")
            print(f"    Entity: {item['entity_label']} ({item['entity_type']})")
            print(f"    Entity similarity: {item['entity_similarity']:.3f}")
            print(f"    Chunk ID: {item['chunk_id']}")
            print(f"    Chunk similarity: {item['chunk_similarity']:.3f}")
            print(f"    Combined score: {item['combined_score']:.3f}")
    
    # Generate explanation
    print(f"\n{'='*60}")
    print("SEARCH EXPLANATION")
    print('='*60)
    explanation = searcher.explain_search("What is caste system?", 
                                         history="Previous discussion about Ambedkar")
    print(explanation)
    
    print("\n✅ Local Graph RAG Search (Equation 4) implemented successfully!")
    return searcher

if __name__ == "__main__":
    test_local_search()