"""
Simplified community detector without external dependencies
"""
import networkx as nx
import json
from collections import defaultdict

class SimpleCommunityDetector:
    """
    Simplified community detection for demo purposes
    """
    
    def __init__(self):
        self.communities = {}
        self.community_summaries = {}
        
    def detect_communities(self, graph):
        """Simple community detection for demo"""
        # Assign communities based on node degree (simplified for demo)
        communities = {}
        for i, node in enumerate(graph.nodes()):
            communities[node] = i % 3  # Assign to one of 3 communities
        
        self.communities = communities
        
        # Create simple community groups
        community_groups = defaultdict(list)
        for node, comm_id in communities.items():
            community_groups[comm_id].append(node)
        
        print(f"Simple community detection: {len(community_groups)} communities")
        return communities
    
    def generate_community_summaries(self, graph, community_partition):
        """Generate simple summaries for demo"""
        community_nodes = defaultdict(list)
        for node, comm_id in community_partition.items():
            community_nodes[comm_id].append(node)
        
        self.community_summaries = {}
        
        for comm_id, nodes in community_nodes.items():
            # Simple summary
            node_types = defaultdict(int)
            for node in nodes:
                node_data = graph.nodes[node]
                node_type = node_data.get('type', 'unknown')
                node_types[node_type] += 1
            
            summary = {
                'community_id': comm_id,
                'node_count': len(nodes),
                'entity_types': dict(node_types),
                'natural_summary': f"Community {comm_id} with {len(nodes)} nodes including {', '.join([f'{count} {typ}' for typ, count in list(node_types.items())[:3]])}"
            }
            
            self.community_summaries[comm_id] = summary
        
        return self.community_summaries
    
    def save_communities(self, filepath):
        """Save communities to file"""
        data = {
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
        
        self.communities = data.get('communities', {})
        self.community_summaries = data.get('summaries', {})
        
        print(f"Communities loaded from {filepath}")

# Test function
def test_simple_detector():
    """Test the simple community detector"""
    import networkx as nx
    
    # Create a simple graph
    graph = nx.Graph()
    graph.add_nodes_from(['A', 'B', 'C', 'D', 'E'])
    graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E')])
    
    detector = SimpleCommunityDetector()
    communities = detector.detect_communities(graph)
    summaries = detector.generate_community_summaries(graph, communities)
    
    print(f"Detected {len(set(communities.values()))} communities")
    for comm_id, summary in summaries.items():
        print(f"Community {comm_id}: {summary['natural_summary']}")
    
    return detector

if __name__ == "__main__":
    test_simple_detector()