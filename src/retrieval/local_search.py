"""
Local Graph RAG Search (Equation 4 from SemRAG Paper)
D_retrieved = Top_k({v ∈ V, g ∈ G | sim(v, Q+H) > τ_e AND sim(g, v) > τ_d})
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json

class LocalGraphSearch:
    """Implements Local Graph RAG Search (Equation 4)"""
    
    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 entity_threshold: float = 0.3,
                 document_threshold: float = 0.4,
                 top_k: int = 5):
        
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.entity_threshold = entity_threshold
        self.document_threshold = document_threshold
        self.top_k = top_k
        self.graph = None
        self.chunks = None
        self.communities = None
        self.community_summaries = None
        
    def load_resources(self, 
                      graph_path: str,
                      chunks_path: str,
                      communities_path: str = None):
        """Load graph, chunks, and communities"""
        print("Loading resources for Local Graph Search...")
        
        # Load graph
        with open(graph_path, 'rb') as f:
            graph_data = pickle.load(f)
        self.graph = graph_data["graph"]
        print(f"  Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        # Load chunks
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
            self.chunks = chunks_data["chunks"]
        print(f"  Chunks loaded: {len(self.chunks)} chunks")
        
        # Load communities if provided
        if communities_path:
            with open(communities_path, 'r', encoding='utf-8') as f:
                communities_data = json.load(f)
                self.communities = communities_data["communities"]
                self.community_summaries = communities_data["summaries"]
            print(f"  Communities loaded: {len(self.communities)} communities")
        
        # Precompute chunk embeddings
        print("  Computing chunk embeddings...")
        self._precompute_chunk_embeddings()
        
        print("Resources loaded successfully")
    
    def _precompute_chunk_embeddings(self):
        """Precompute embeddings for all chunks"""
        self.chunk_embeddings = {}
        self.chunk_texts = {}
        
        for chunk in self.chunks:
            chunk_id = chunk.get("id", 0)
            chunk_text = chunk.get("text", "")
            
            if chunk_text:
                embedding = self.embedding_model.encode([chunk_text])[0]
                self.chunk_embeddings[chunk_id] = embedding
                self.chunk_texts[chunk_id] = chunk_text
    
    def search(self, query: str, history: str = "") -> Dict[str, Any]:
        """Perform Local Graph Search (Equation 4)"""
        print("\n" + "="*60)
        print("LOCAL GRAPH RAG SEARCH (Equation 4)")
        print("="*60)
        
        # Combine query and history
        query_with_history = query + " " + history if history else query
        query_embedding = self.embedding_model.encode([query_with_history])[0]
        
        # Step 1: Find relevant entities (v ∈ V where sim(v, Q+H) > τ_e)
        relevant_entities = self._find_relevant_entities(query_embedding)
        print(f"Found {len(relevant_entities)} relevant entities (threshold: {self.entity_threshold})")
        
        # Step 2: Find relevant documents (g ∈ G where sim(g, v) > τ_d)
        relevant_chunks = self._find_relevant_chunks(relevant_entities, query_embedding)
        print(f"Found {len(relevant_chunks)} relevant chunks (threshold: {self.document_threshold})")
        
        # Step 3: Get top-k chunks
        retrieved_chunks = self._get_top_k_chunks(relevant_chunks)
        
        # Prepare results
        results = {
            "query": query,
            "history": history,
            "relevant_entities": relevant_entities[:10],  # Top 10 entities
            "retrieved_chunks": retrieved_chunks,
            "search_metrics": {
                "entity_threshold": self.entity_threshold,
                "document_threshold": self.document_threshold,
                "top_k": self.top_k,
                "relevant_entities_count": len(relevant_entities),
                "relevant_chunks_count": len(relevant_chunks)
            }
        }
        
        return results
    
    def _find_relevant_entities(self, query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """Find entities similar to query (sim(v, Q+H) > τ_e)"""
        relevant_entities = []
        
        for node_id in self.graph.nodes():
            node = self.graph.nodes[node_id]
            
            if "embedding" in node:
                entity_embedding = node["embedding"]
                similarity = cosine_similarity([query_embedding], [entity_embedding])[0][0]
                
                if similarity > self.entity_threshold:
                    relevant_entities.append({
                        "node_id": node_id,
                        "text": node.get("text", ""),
                        "type": node.get("type", ""),
                        "similarity": float(similarity),
                        "community": node.get("community", -1)
                    })
        
        # Sort by similarity (descending)
        relevant_entities.sort(key=lambda x: x["similarity"], reverse=True)
        return relevant_entities
    
    def _find_relevant_chunks(self, relevant_entities: List[Dict], 
                            query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """Find chunks relevant to entities (sim(g, v) > τ_d)"""
        relevant_chunks = {}
        
        for entity in relevant_entities[:50]:  # Limit to top 50 entities
            node_id = entity["node_id"]
            chunk_ids = self.graph.nodes[node_id].get("chunk_ids", [])
            
            for chunk_id in chunk_ids:
                if chunk_id in self.chunk_embeddings:
                    chunk_embedding = self.chunk_embeddings[chunk_id]
                    
                    # Check similarity with entity
                    entity_similarity = entity["similarity"]
                    
                    # Check similarity with query
                    query_similarity = cosine_similarity(
                        [query_embedding], [chunk_embedding]
                    )[0][0]
                    
                    # Combined score (weighted average)
                    combined_score = 0.6 * query_similarity + 0.4 * entity_similarity
                    
                    if combined_score > self.document_threshold:
                        if chunk_id not in relevant_chunks or combined_score > relevant_chunks[chunk_id]["score"]:
                            relevant_chunks[chunk_id] = {
                                "chunk_id": chunk_id,
                                "text": self.chunk_texts[chunk_id],
                                "score": float(combined_score),
                                "query_similarity": float(query_similarity),
                                "entity_similarity": float(entity_similarity),
                                "connected_entities": [
                                    {
                                        "text": entity["text"],
                                        "type": entity["type"],
                                        "similarity": entity["similarity"]
                                    }
                                ]
                            }
                        else:
                            # Add entity to existing chunk
                            relevant_chunks[chunk_id]["connected_entities"].append({
                                "text": entity["text"],
                                "type": entity["type"],
                                "similarity": entity["similarity"]
                            })
        
        # Convert to list
        chunks_list = list(relevant_chunks.values())
        
        # Sort by score (descending)
        chunks_list.sort(key=lambda x: x["score"], reverse=True)
        return chunks_list
    
    def _get_top_k_chunks(self, relevant_chunks: List[Dict]) -> List[Dict[str, Any]]:
        """Get top-k chunks"""
        top_chunks = relevant_chunks[:self.top_k]
        
        # Format for output
        formatted_chunks = []
        for chunk in top_chunks:
            formatted_chunks.append({
                "id": chunk["chunk_id"],
                "text": chunk["text"][:500] + "..." if len(chunk["text"]) > 500 else chunk["text"],
                "score": chunk["score"],
                "entities": [
                    {
                        "name": entity["text"],
                        "type": entity["type"]
                    }
                    for entity in chunk["connected_entities"][:3]  # Top 3 entities
                ]
            })
        
        return formatted_chunks
    
    def print_results(self, results: Dict[str, Any]):
        """Print search results in readable format"""
        print(f"\nQUERY: {results['query']}")
        if results['history']:
            print(f"HISTORY: {results['history']}")
        
        print(f"\nRELEVANT ENTITIES (Top {len(results['relevant_entities'])}):")
        for i, entity in enumerate(results['relevant_entities'][:5], 1):
            print(f"  {i}. {entity['text']} ({entity['type']}) - Similarity: {entity['similarity']:.3f}")
        
        print(f"\nRETRIEVED CHUNKS (Top {len(results['retrieved_chunks'])}):")
        for i, chunk in enumerate(results['retrieved_chunks'], 1):
            print(f"\n  CHUNK {i} (Score: {chunk['score']:.3f}):")
            print(f"  Text: {chunk['text'][:200]}...")
            print(f"  Entities: {', '.join([e['name'] for e in chunk['entities'][:2]])}")
        
        metrics = results['search_metrics']
        print(f"\nSEARCH METRICS:")
        print(f"  Entity threshold (τ_e): {metrics['entity_threshold']}")
        print(f"  Document threshold (τ_d): {metrics['document_threshold']}")
        print(f"  Relevant entities: {metrics['relevant_entities_count']}")
        print(f"  Relevant chunks: {metrics['relevant_chunks_count']}")

def main():
    """Test Local Graph Search"""
    # Initialize searcher
    searcher = LocalGraphSearch(
        entity_threshold=0.3,
        document_threshold=0.4,
        top_k=3
    )
    
    # Load resources
    searcher.load_resources(
        graph_path="../../data/processed/knowledge_graph.pkl",
        chunks_path="../../data/processed/chunks.json",
        communities_path="../../data/processed/communities.json"
    )
    
    # Test queries
    test_queries = [
        "What is caste according to Ambedkar?",
        "How does Ambedkar view education?",
        "What are the fundamental rights mentioned by Ambedkar?",
        "What is the relationship between Hinduism and Buddhism according to Ambedkar?"
    ]
    
    for query in test_queries:
        print("\n" + "="*60)
        print(f"TESTING QUERY: {query}")
        print("="*60)
        
        # Perform search
        results = searcher.search(query)
        
        # Print results
        searcher.print_results(results)

if __name__ == "__main__":
    main()