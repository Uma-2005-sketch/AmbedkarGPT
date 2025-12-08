"""
Knowledge graph construction for SEMRAG
"""
import networkx as nx
import json
from collections import defaultdict

class KnowledgeGraphBuilder:
    """
    Builds knowledge graph from extracted entities and relations
    """
    
    def __init__(self):
        self.graph = nx.Graph()
        self.entity_nodes = {}
        self.chunk_nodes = {}
        
    def add_entities(self, entities, chunk_id=None):
        """Add entities as nodes to the graph"""
        for entity in entities:
            node_id = f"{entity['text']}_{entity['type']}"
            
            if node_id not in self.entity_nodes:
                self.graph.add_node(
                    node_id,
                    label=entity['text'],
                    type=entity['type'],
                    entity_type=entity['label']
                )
                self.entity_nodes[node_id] = entity
            
            # Connect entity to chunk if chunk_id provided
            if chunk_id is not None:
                chunk_node_id = f"chunk_{chunk_id}"
                if chunk_node_id not in self.chunk_nodes:
                    self.graph.add_node(chunk_node_id, type='chunk', chunk_id=chunk_id)
                    self.chunk_nodes[chunk_node_id] = chunk_id
                
                # Add edge between entity and chunk
                self.graph.add_edge(node_id, chunk_node_id, 
                                   relation='mentioned_in', weight=1)
    
    def add_relations(self, relations):
        """Add relations as edges between entities"""
        for rel in relations:
            subject_id = f"{rel['subject']}_person"
            object_id = f"{rel['object']}_{self._get_entity_type(rel['object'])}"
            
            # Ensure nodes exist
            if subject_id not in self.graph.nodes():
                self.graph.add_node(subject_id, label=rel['subject'], type='person')
            
            if object_id not in self.graph.nodes():
                entity_type = self._get_entity_type(rel['object'])
                self.graph.add_node(object_id, label=rel['object'], type=entity_type)
            
            # Add relation edge
            if self.graph.has_edge(subject_id, object_id):
                # Increase weight if relation exists
                self.graph[subject_id][object_id]['weight'] += 1
                self.graph[subject_id][object_id]['relations'].append(rel['relation'])
            else:
                self.graph.add_edge(
                    subject_id, object_id,
                    relation=rel['relation'],
                    relations=[rel['relation']],
                    weight=1,
                    context=rel.get('context', '')
                )
    
    def _get_entity_type(self, entity_text):
        """Determine entity type from text"""
        entity_text_lower = entity_text.lower()
        
        # Check against common categories
        if any(word in entity_text_lower for word in ['caste', 'equality', 'democracy']):
            return 'concept'
        elif any(word in entity_text_lower for word in ['annihilation', 'constitution']):
            return 'work'
        else:
            return 'concept'  # Default
    
    def get_graph_stats(self):
        """Get statistics about the graph"""
        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'entity_nodes': len(self.entity_nodes),
            'chunk_nodes': len(self.chunk_nodes),
            'node_types': defaultdict(int),
            'edge_relations': defaultdict(int)
        }
        
        # Count node types
        for node, data in self.graph.nodes(data=True):
            stats['node_types'][data.get('type', 'unknown')] += 1
        
        # Count edge relations
        for u, v, data in self.graph.edges(data=True):
            rel = data.get('relation', 'unknown')
            stats['edge_relations'][rel] += 1
        
        return stats
    
    def save_graph(self, filepath):
        """Save graph to JSON file"""
        graph_data = {
            'nodes': [],
            'edges': []
        }
        
        # Add nodes
        for node, data in self.graph.nodes(data=True):
            graph_data['nodes'].append({
                'id': node,
                'label': data.get('label', node),
                'type': data.get('type', 'unknown'),
                'entity_type': data.get('entity_type', '')
            })
        
        # Add edges
        for u, v, data in self.graph.edges(data=True):
            graph_data['edges'].append({
                'source': u,
                'target': v,
                'relation': data.get('relation', ''),
                'weight': data.get('weight', 1),
                'context': data.get('context', '')
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        print(f"Graph saved to {filepath} with {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges")
    
    def load_graph(self, filepath):
        """Load graph from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        self.graph.clear()
        self.entity_nodes.clear()
        self.chunk_nodes.clear()
        
        # Add nodes
        for node_data in graph_data['nodes']:
            self.graph.add_node(
                node_data['id'],
                label=node_data.get('label', ''),
                type=node_data.get('type', ''),
                entity_type=node_data.get('entity_type', '')
            )
            
            if node_data.get('type') == 'chunk':
                self.chunk_nodes[node_data['id']] = node_data.get('chunk_id', 0)
            else:
                self.entity_nodes[node_data['id']] = {
                    'text': node_data.get('label', ''),
                    'type': node_data.get('type', '')
                }
        
        # Add edges
        for edge_data in graph_data['edges']:
            self.graph.add_edge(
                edge_data['source'],
                edge_data['target'],
                relation=edge_data.get('relation', ''),
                weight=edge_data.get('weight', 1),
                context=edge_data.get('context', '')
            )
        
        print(f"Graph loaded from {filepath}")

# Test function
def test_graph_builder():
    """Test knowledge graph construction"""
    from entity_extractor import EntityExtractor
    
    extractor = EntityExtractor()
    graph_builder = KnowledgeGraphBuilder()
    
    test_texts = [
        "Ambedkar criticized the caste system in Annihilation of Caste.",
        "He advocated for equality and democracy in Indian Constitution.",
        "Gandhi had different views on the caste system."
    ]
    
    print("KNOWLEDGE GRAPH CONSTRUCTION TEST")
    print("=" * 50)
    
    for i, text in enumerate(test_texts):
        print(f"\nProcessing chunk {i+1}: {text[:50]}...")
        
        # Extract entities
        entities = extractor.extract_entities(text)
        relations = extractor.extract_relations(text, entities)
        
        # Add to graph
        graph_builder.add_entities(entities, chunk_id=i)
        if relations:
            graph_builder.add_relations(relations)
    
    # Get stats
    stats = graph_builder.get_graph_stats()
    print(f"\nGraph Statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Total edges: {stats['total_edges']}")
    print(f"  Entity nodes: {stats['entity_nodes']}")
    print(f"  Chunk nodes: {stats['chunk_nodes']}")
    
    # Save graph
    graph_builder.save_graph("test_knowledge_graph.json")
    
    print("\n✅ Knowledge graph construction complete!")
    return graph_builder

if __name__ == "__main__":
    test_graph_builder()