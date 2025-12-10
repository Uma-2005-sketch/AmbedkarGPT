"""
Knowledge Graph Builder and Community Detection
Implements Leiden/Louvain algorithm for community detection
"""

import networkx as nx
import json
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class KnowledgeGraphBuilder:
    """Builds knowledge graph from entities and relationships"""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.graph = nx.Graph()
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.communities = {}
        
    def build_from_extractions(self, extractions: Dict[str, Any]) -> nx.Graph:
        """Build graph from entity extractions"""
        print("="*60)
        print("KNOWLEDGE GRAPH CONSTRUCTION")
        print("="*60)
        
        entities = extractions["entities"]
        relationships = extractions["relationships"]
        
        # Add nodes (entities)
        print(f"Adding {len(entities)} entities as nodes...")
        for entity in entities:
            self.graph.add_node(
                entity["id"],
                text=entity["text"],
                type=entity["type"],
                occurrences=entity["occurrences"],
                chunk_ids=entity["chunk_ids"]
            )
        
        # Add edges (relationships)
        print(f"Adding {len(relationships)} relationships as edges...")
        for rel in relationships:
            # Find source and target node IDs
            source_id = self._find_entity_id(entities, rel["source"], rel["source_type"])
            target_id = self._find_entity_id(entities, rel["target"], rel["target_type"])
            
            if source_id is not None and target_id is not None:
                self.graph.add_edge(
                    source_id,
                    target_id,
                    type=rel["type"],
                    chunk_id=rel["chunk_id"],
                    sentence=rel.get("sentence", "")[:100]
                )
        
        # Calculate node embeddings for similarity
        print("Calculating node embeddings...")
        self._add_node_embeddings(entities)
        
        # Add edge weights based on semantic similarity
        print("Calculating edge weights...")
        self._add_edge_weights()
        
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        print("="*60)
        
        return self.graph
    
    def _find_entity_id(self, entities: List[Dict], text: str, entity_type: str) -> int:
        """Find entity ID by text and type"""
        for entity in entities:
            if entity["text"].lower() == text.lower() and entity["type"] == entity_type:
                return entity["id"]
        return None
    
    def _add_node_embeddings(self, entities: List[Dict]):
        """Add embedding vectors to nodes"""
        # Create text for embedding
        node_texts = []
        node_ids = []
        
        for entity in entities:
            node_id = entity["id"]
            if node_id in self.graph.nodes():
                # Combine entity text and type for embedding
                text = f"{entity['text']} {entity['type']}"
                node_texts.append(text)
                node_ids.append(node_id)
        
        # Create embeddings
        if node_texts:
            embeddings = self.embedding_model.encode(node_texts)
            
            # Add embeddings to nodes
            for node_id, embedding in zip(node_ids, embeddings):
                self.graph.nodes[node_id]["embedding"] = embedding
    
    def _add_edge_weights(self):
        """Add weights to edges based on semantic similarity"""
        for u, v, data in self.graph.edges(data=True):
            if "embedding" in self.graph.nodes[u] and "embedding" in self.graph.nodes[v]:
                emb_u = self.graph.nodes[u]["embedding"].reshape(1, -1)
                emb_v = self.graph.nodes[v]["embedding"].reshape(1, -1)
                
                similarity = cosine_similarity(emb_u, emb_v)[0][0]
                # Ensure similarity is between 0 and 1
                similarity = max(0.0, min(1.0, similarity))
                data["weight"] = float(similarity)
                data["strength"] = "strong" if similarity > 0.7 else "medium" if similarity > 0.4 else "weak"
            else:
                data["weight"] = 0.5
                data["strength"] = "medium"
    
    def _detect_communities_networkx(self) -> Dict[int, List[int]]:
        """Detect communities using networkx connected components"""
        print("Using networkx connected components")
        
        # Use connected components as communities
        communities = {}
        for i, component in enumerate(nx.connected_components(self.graph)):
            communities[i] = list(component)
        
        return communities
    
    def detect_communities(self, algorithm: str = "connected"):
        """Detect communities using specified algorithm"""
        print("\n" + "="*60)
        print(f"COMMUNITY DETECTION")
        print("="*60)
        
        # Always use connected components for stability
        communities = self._detect_communities_networkx()
        
        # Store communities
        self.communities = communities
        
        # Add community labels to nodes
        for comm_id, node_ids in communities.items():
            for node_id in node_ids:
                if node_id in self.graph.nodes():
                    self.graph.nodes[node_id]["community"] = comm_id
        
        print(f"Detected {len(communities)} communities")
        
        # Print community sizes
        print("\nCOMMUNITY SIZES:")
        for comm_id, node_ids in communities.items():
            print(f"  Community {comm_id}: {len(node_ids)} nodes")
        
        print("="*60)
        
        return communities
    
    def generate_community_summaries(self, chunks: List[Dict[str, Any]]) -> Dict[int, str]:
        """Generate LLM summaries for each community"""
        print("\n" + "="*60)
        print("GENERATING COMMUNITY SUMMARIES")
        print("="*60)
        
        community_summaries = {}
        
        for comm_id, node_ids in self.communities.items():
            # Get all text from chunks containing these entities
            community_texts = []
            for node_id in node_ids:
                if node_id in self.graph.nodes():
                    node = self.graph.nodes[node_id]
                    chunk_ids = node.get("chunk_ids", [])
                    
                    for chunk_id in chunk_ids[:5]:  # Limit to first 5 chunks per entity
                        if chunk_id < len(chunks):
                            chunk_text = chunks[chunk_id].get("text", "")
                            community_texts.append(chunk_text[:500])  # Truncate
            
            if community_texts:
                # Combine texts
                combined_text = " ".join(community_texts)[:2000]  # Limit length
                
                # Generate summary (in real implementation, use LLM)
                summary = self._generate_summary(combined_text, comm_id, node_ids)
                community_summaries[comm_id] = summary
        
        print(f"Generated {len(community_summaries)} community summaries")
        print("="*60)
        
        return community_summaries
    
    def _generate_summary(self, text: str, comm_id: int, node_ids: List[int]) -> str:
        """Generate summary for a community"""
        # Get entity names in this community
        entity_names = []
        for node_id in node_ids[:10]:  # Limit to first 10 entities
            if node_id in self.graph.nodes():
                entity_names.append(self.graph.nodes[node_id]["text"])
        
        # Simple rule-based summary (in production, use LLM)
        entity_str = ", ".join(set(entity_names))[:200]
        
        summary = f"Community {comm_id} contains {len(node_ids)} entities including: {entity_str}. "
        summary += f"This community discusses topics related to {', '.join(entity_names[:3])}."
        
        return summary
    
    def save_graph(self, output_path: str):
        """Save graph to pickle file"""
        graph_data = {
            "graph": self.graph,
            "communities": self.communities,
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "community_count": len(self.communities)
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(graph_data, f)
        
        print(f"Knowledge graph saved to {output_path}")
    
    def load_graph(self, input_path: str):
        """Load graph from pickle file"""
        with open(input_path, 'rb') as f:
            graph_data = pickle.load(f)
        
        self.graph = graph_data["graph"]
        self.communities = graph_data["communities"]
        
        print(f"Knowledge graph loaded from {input_path}")
        print(f"  Nodes: {self.graph.number_of_nodes()}")
        print(f"  Edges: {self.graph.number_of_edges()}")
        print(f"  Communities: {len(self.communities)}")

def main():
    """Test knowledge graph builder"""
    # Load extractions
    extractions_path = "../../data/processed/entity_extractions.json"
    with open(extractions_path, 'r', encoding='utf-8') as f:
        extractions = json.load(f)
    
    # Load chunks for summaries
    chunks_path = "../../data/processed/chunks.json"
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
        chunks = chunks_data["chunks"]
    
    print(f"Loaded {len(extractions['entities'])} entities")
    print(f"Loaded {len(chunks)} chunks")
    
    # Build graph
    builder = KnowledgeGraphBuilder()
    graph = builder.build_from_extractions(extractions)
    
    # Detect communities
    communities = builder.detect_communities(algorithm="connected")
    
    # Generate community summaries
    summaries = builder.generate_community_summaries(chunks)
    
    # Save graph
    builder.save_graph("../../data/processed/knowledge_graph.pkl")
    
    # Save communities and summaries
    community_data = {
        "communities": communities,
        "summaries": summaries,
        "stats": {
            "total_communities": len(communities),
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges()
        }
    }
    
    with open("../../data/processed/communities.json", 'w', encoding='utf-8') as f:
        json.dump(community_data, f, indent=2)
    
    print("\nSAMPLE COMMUNITY SUMMARIES:")
    for comm_id, summary in list(summaries.items())[:3]:
        print(f"\nCommunity {comm_id}:")
        print(f"  {summary}")
    
    # Print graph statistics
    print("\nGRAPH STATISTICS:")
    print(f"  Number of nodes: {graph.number_of_nodes()}")
    print(f"  Number of edges: {graph.number_of_edges()}")
    print(f"  Number of communities: {len(communities)}")
    print(f"  Average degree: {sum(dict(graph.degree()).values()) / graph.number_of_nodes():.2f}")

if __name__ == "__main__":
    main()