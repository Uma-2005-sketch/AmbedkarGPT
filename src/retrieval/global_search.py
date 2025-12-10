"""
Global Graph RAG Search (Equation 5 from SemRAG Paper)
Finds relevant community summaries and retrieves chunks from those communities
"""

import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

class GlobalGraphSearch:
    """Implements Global Graph RAG Search (Equation 5)"""
    
    def __init__(self,
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 community_threshold: float = 0.4,
                 top_k_communities: int = 3,
                 top_k_chunks: int = 5):
        
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.community_threshold = community_threshold
        self.top_k_communities = top_k_communities
        self.top_k_chunks = top_k_chunks
        self.communities = None
        self.community_summaries = None
        self.chunks = None
        self.community_embeddings = {}
        
    def load_resources(self, 
                      communities_path: str,
                      chunks_path: str):
        """Load communities and chunks"""
        print("Loading resources for Global Graph Search...")
        
        # Load communities
        with open(communities_path, 'r', encoding='utf-8') as f:
            communities_data = json.load(f)
            self.communities = communities_data["communities"]
            self.community_summaries = communities_data["summaries"]
        print(f"  Communities loaded: {len(self.communities)} communities")
        
        # Load chunks
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
            self.chunks = chunks_data["chunks"]
        print(f"  Chunks loaded: {len(self.chunks)} chunks")
        
        # Precompute community summary embeddings
        print("  Computing community embeddings...")
        self._precompute_community_embeddings()
        
        # Create chunk-community mapping
        print("  Creating chunk-community mapping...")
        self.chunk_communities = self._create_chunk_community_mapping()
        
        print("Resources loaded successfully")
    
    def _precompute_community_embeddings(self):
        """Precompute embeddings for community summaries"""
        for comm_id, summary in self.community_summaries.items():
            embedding = self.embedding_model.encode([summary])[0]
            self.community_embeddings[int(comm_id)] = embedding
    
    def _create_chunk_community_mapping(self) -> Dict[int, List[int]]:
        """Map chunks to their communities"""
        chunk_communities = {}
        
        for comm_id, node_ids in self.communities.items():
            for node_id in node_ids:
                # Get chunk IDs from node (in real implementation, nodes would have chunk references)
                # For now, we'll map based on chunk ID modulo number of communities
                for chunk_id in range(len(self.chunks)):
                    if chunk_id not in chunk_communities:
                        chunk_communities[chunk_id] = []
                    if int(comm_id) not in chunk_communities[chunk_id]:
                        chunk_communities[chunk_id].append(int(comm_id))
        
        return chunk_communities
    
    def search(self, query: str) -> Dict[str, Any]:
        """Perform Global Graph Search (Equation 5)"""
        print("\n" + "="*60)
        print("GLOBAL GRAPH RAG SEARCH (Equation 5)")
        print("="*60)
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Step 1: Find relevant communities
        relevant_communities = self._find_relevant_communities(query_embedding)
        print(f"Found {len(relevant_communities)} relevant communities (threshold: {self.community_threshold})")
        
        # Step 2: Get chunks from relevant communities
        relevant_chunks = self._get_chunks_from_communities(relevant_communities, query_embedding)
        print(f"Found {len(relevant_chunks)} chunks from relevant communities")
        
        # Step 3: Get top-k chunks
        retrieved_chunks = relevant_chunks[:self.top_k_chunks]
        
        # Prepare results
        results = {
            "query": query,
            "relevant_communities": relevant_communities[:self.top_k_communities],
            "retrieved_chunks": retrieved_chunks,
            "search_metrics": {
                "community_threshold": self.community_threshold,
                "top_k_communities": self.top_k_communities,
                "top_k_chunks": self.top_k_chunks,
                "relevant_communities_count": len(relevant_communities),
                "relevant_chunks_count": len(relevant_chunks)
            }
        }
        
        return results
    
    def _find_relevant_communities(self, query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """Find communities similar to query"""
        relevant_communities = []
        
        for comm_id, comm_embedding in self.community_embeddings.items():
            similarity = cosine_similarity([query_embedding], [comm_embedding])[0][0]
            
            if similarity > self.community_threshold:
                summary = self.community_summaries.get(str(comm_id), "No summary available")
                node_count = len(self.communities.get(str(comm_id), []))
                
                relevant_communities.append({
                    "community_id": comm_id,
                    "similarity": float(similarity),
                    "summary": summary[:200] + "..." if len(summary) > 200 else summary,
                    "node_count": node_count
                })
        
        # Sort by similarity (descending)
        relevant_communities.sort(key=lambda x: x["similarity"], reverse=True)
        return relevant_communities
    
    def _get_chunks_from_communities(self, 
                                   relevant_communities: List[Dict],
                                   query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """Get chunks from relevant communities"""
        relevant_chunks = {}
        
        for community in relevant_communities[:self.top_k_communities]:
            comm_id = community["community_id"]
            
            # Get chunks associated with this community
            for chunk_id, chunk_communities in self.chunk_communities.items():
                if comm_id in chunk_communities and chunk_id < len(self.chunks):
                    chunk = self.chunks[chunk_id]
                    chunk_text = chunk.get("text", "")
                    
                    if chunk_text:
                        # Compute chunk embedding and similarity
                        chunk_embedding = self.embedding_model.encode([chunk_text])[0]
                        similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
                        
                        # Combined score (community similarity + chunk similarity)
                        combined_score = 0.5 * community["similarity"] + 0.5 * similarity
                        
                        if chunk_id not in relevant_chunks or combined_score > relevant_chunks[chunk_id]["score"]:
                            relevant_chunks[chunk_id] = {
                                "chunk_id": chunk_id,
                                "text": chunk_text,
                                "score": float(combined_score),
                                "community_similarity": community["similarity"],
                                "chunk_similarity": float(similarity),
                                "communities": [comm_id]
                            }
                        else:
                            # Add community to existing chunk
                            if comm_id not in relevant_chunks[chunk_id]["communities"]:
                                relevant_chunks[chunk_id]["communities"].append(comm_id)
        
        # Convert to list and sort
        chunks_list = list(relevant_chunks.values())
        chunks_list.sort(key=lambda x: x["score"], reverse=True)
        return chunks_list
    
    def print_results(self, results: Dict[str, Any]):
        """Print search results in readable format"""
        print(f"\nQUERY: {results['query']}")
        
        print(f"\nRELEVANT COMMUNITIES (Top {len(results['relevant_communities'])}):")
        for i, community in enumerate(results['relevant_communities'], 1):
            print(f"\n  COMMUNITY {i} (ID: {community['community_id']}, Similarity: {community['similarity']:.3f}):")
            print(f"  Summary: {community['summary']}")
            print(f"  Nodes: {community['node_count']}")
        
        print(f"\nRETRIEVED CHUNKS (Top {len(results['retrieved_chunks'])}):")
        for i, chunk in enumerate(results['retrieved_chunks'], 1):
            print(f"\n  CHUNK {i} (Score: {chunk['score']:.3f}):")
            print(f"  Text: {chunk['text'][:200]}...")
            print(f"  From communities: {chunk['communities'][:3]}")
        
        metrics = results['search_metrics']
        print(f"\nSEARCH METRICS:")
        print(f"  Community threshold: {metrics['community_threshold']}")
        print(f"  Top-K communities: {metrics['top_k_communities']}")
        print(f"  Top-K chunks: {metrics['top_k_chunks']}")
        print(f"  Relevant communities: {metrics['relevant_communities_count']}")
        print(f"  Relevant chunks: {metrics['relevant_chunks_count']}")

def main():
    """Test Global Graph Search"""
    # Initialize searcher
    searcher = GlobalGraphSearch(
        community_threshold=0.4,
        top_k_communities=2,
        top_k_chunks=3
    )
    
    # Load resources
    searcher.load_resources(
        communities_path="../../data/processed/communities.json",
        chunks_path="../../data/processed/chunks.json"
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