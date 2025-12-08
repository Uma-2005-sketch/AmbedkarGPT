"""
Community detection for knowledge graph (Leiden/Louvain algorithms)
"""
import networkx as nx
try:
    import community as community_louvain  # python-louvain package
except ImportError:
    # Try alternative import
    import python_louvain as community_louvain
try:
    import leidenalg
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    print("Note: leidenalg not installed. Using Louvain algorithm.")
import json
from collections import defaultdict

class CommunityDetector:
    """
    Detect communities in knowledge graph using Leiden or Louvain algorithms
    """
    
    def __init__(self, method='louvain'):
        """
        Args:
            method: 'louvain' or 'leiden'
        """
        self.method = method if method in ['louvain', 'leiden'] else 'louvain'
        self.communities = {}
        self.community_summaries = {}
        
    def detect_communities(self, graph):
        """
        Detect communities in the graph
        
        Args:
            graph: networkx Graph
            
        Returns:
            Dictionary mapping node -> community_id
        """
        if self.method == 'leiden' and LEIDEN_AVAILABLE:
            return self._detect_leiden(graph)
        else:
            return self._detect_louvain(graph)
    
    def _detect_louvain(self, graph):
        """Use Louvain algorithm for community detection"""
        # Convert to undirected weighted graph
        undirected_graph = graph.to_undirected()
        
        # Detect communities
        partition = community_louvain.best_partition(undirected_graph)
        
        # Store communities
        self.communities = partition
        
        # Group nodes by community
        community_groups = defaultdict(list)
        for node, comm_id in partition.items():
            community_groups[comm_id].append(node)
        
        print(f"Louvain detected {len(community_groups)} communities")
        return partition
    
    def _detect_leiden(self, graph):
        """Use Leiden algorithm for community detection"""
        try:
            import igraph as ig
            
            # Convert networkx to igraph
            edges = list(graph.edges(data=True))
            ig_graph = ig.Graph()
            
            # Add vertices
            ig_graph.add_vertices(list(graph.nodes()))
            
            # Add edges with weights
            edge_list = []
            weights = []
            for u, v, data in edges:
                edge_list.append((u, v))
                weights.append(data.get('weight', 1))
            
            ig_graph.add_edges(edge_list)
            ig_graph.es['weight'] = weights
            
            # Detect communities using Leiden algorithm
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.ModularityVertexPartition,
                weights=weights
            )
            
            # Convert to dictionary
            communities = {}
            for i, community in enumerate(partition):
                for node_idx in community:
                    node_name = ig_graph.vs[node_idx]['name']
                    communities[node_name] = i
            
            self.communities = communities
            community_groups = defaultdict(list)
            for node, comm_id in communities.items():
                community_groups[comm_id].append(node)
            
            print(f"Leiden detected {len(community_groups)} communities")
            return communities
            
        except Exception as e:
            print(f"Leiden failed: {e}. Falling back to Louvain.")
            return self._detect_louvain(graph)
    
    def generate_community_summaries(self, graph, community_partition):
        """
        Generate summaries for each community
        
        Args:
            graph: networkx Graph
            community_partition: dict mapping node -> community_id
            
        Returns:
            dict mapping community_id -> summary
        """
        # Group nodes by community
        community_nodes = defaultdict(list)
        for node, comm_id in community_partition.items():
            community_nodes[comm_id].append(node)
        
        self.community_summaries = {}
        
        for comm_id, nodes in community_nodes.items():
            # Extract community subgraph
            subgraph = graph.subgraph(nodes)
            
            # Get community statistics
            summary = self._summarize_community(subgraph, comm_id, nodes)
            self.community_summaries[comm_id] = summary
        
        return self.community_summaries
    
    def _summarize_community(self, subgraph, comm_id, nodes):
        """Generate summary for a community"""
        # Count entity types
        entity_types = defaultdict(int)
        central_nodes = []
        
        for node in nodes:
            node_data = subgraph.nodes[node]
            entity_type = node_data.get('type', 'unknown')
            entity_types[entity_type] += 1
            
            # Calculate centrality (simplified)
            if subgraph.degree(node) > 0:
                central_nodes.append({
                    'node': node,
                    'label': node_data.get('label', node),
                    'degree': subgraph.degree(node),
                    'type': entity_type
                })
        
        # Sort by degree
        central_nodes.sort(key=lambda x: x['degree'], reverse=True)
        
        # Create summary
        summary = {
            'community_id': comm_id,
            'node_count': len(nodes),
            'edge_count': subgraph.number_of_edges(),
            'entity_types': dict(entity_types),
            'top_nodes': central_nodes[:5],  # Top 5 central nodes
            'main_themes': self._extract_themes(subgraph, central_nodes)
        }
        
        # Generate natural language summary
        summary['natural_summary'] = self._generate_natural_summary(summary)
        
        return summary
    
    def _extract_themes(self, subgraph, central_nodes):
        """Extract main themes from community"""
        themes = []
        
        # Look at edge relations
        edge_relations = defaultdict(int)
        for u, v, data in subgraph.edges(data=True):
            relation = data.get('relation', '')
            if relation:
                edge_relations[relation] += 1
        
        # Top relations
        top_relations = sorted(edge_relations.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Top node labels
        top_node_labels = [node['label'] for node in central_nodes[:3]]
        
        themes.extend(top_relations)
        themes.extend(top_node_labels)
        
        return themes[:5]  # Return top 5 themes
    
    def _generate_natural_summary(self, summary):
        """Generate natural language summary of community"""
        themes = summary.get('main_themes', [])
        top_nodes = summary.get('top_nodes', [])
        
        if not themes and not top_nodes:
            return f"Community {summary['community_id']} with {summary['node_count']} nodes."
        
        # Extract key information
        theme_text = ", ".join([str(t) for t in themes[:3]])
        node_text = ", ".join([n['label'] for n in top_nodes[:3]])
        
        return (f"Community {summary['community_id']} contains {summary['node_count']} nodes "
                f"focused on {theme_text}. Key entities include {node_text}.")
    
    def save_communities(self, filepath):
        """Save communities and summaries to file"""
        data = {
            'method': self.method,
            'communities': self.communities,
            'summaries': self.community_summaries
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"Communities saved to {filepath}")
    
    def load_communities(self, filepath):
        """Load communities from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.method = data.get('method', 'louvain')
        self.communities = data.get('communities', {})
        self.community_summaries = data.get('summaries', {})
        
        print(f"Communities loaded from {filepath}")

# Test function
def test_community_detection():
    """Test community detection"""
    from graph_builder import KnowledgeGraphBuilder
    from entity_extractor import EntityExtractor
    
    print("COMMUNITY DETECTION TEST")
    print("=" * 50)
    
    # Create a test graph
    extractor = EntityExtractor()
    graph_builder = KnowledgeGraphBuilder()
    
    test_texts = [
        "Ambedkar criticized caste system and advocated equality.",
        "Gandhi had views on caste different from Ambedkar.",
        "Indian Constitution was drafted by Ambedkar.",
        "Buddhism was adopted by Ambedkar in 1956.",
        "Democracy and liberty are key constitutional principles."
    ]
    
    for i, text in enumerate(test_texts):
        entities = extractor.extract_entities(text)
        relations = extractor.extract_relations(text, entities)
        graph_builder.add_entities(entities, chunk_id=i)
        if relations:
            graph_builder.add_relations(relations)
    
    graph = graph_builder.graph
    print(f"Test graph created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Detect communities
    detector = CommunityDetector(method='louvain')
    communities = detector.detect_communities(graph)
    
    # Generate summaries
    summaries = detector.generate_community_summaries(graph, communities)
    
    # Display results
    print(f"\nDetected {len(set(communities.values()))} communities:")
    print("-" * 50)
    
    for comm_id in sorted(set(communities.values())):
        summary = summaries.get(comm_id, {})
        print(f"\nCommunity {comm_id}:")
        print(f"  Nodes: {summary.get('node_count', 0)}")
        print(f"  Main themes: {summary.get('main_themes', [])[:3]}")
        print(f"  Summary: {summary.get('natural_summary', '')[:100]}...")
    
    # Save communities
    detector.save_communities("test_communities.json")
    
    print("\n✅ Community detection complete!")
    return detector

if __name__ == "__main__":
    test_community_detection()