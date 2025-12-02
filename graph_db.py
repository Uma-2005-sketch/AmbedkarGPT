"""
Graph Database Interface for AmbedkarGPT
Mock implementation - can be replaced with Neo4j or AWS Neptune later.
"""

import json
import os
from typing import List, Dict, Tuple

class MockGraphDB:
    """Mock graph database for local development"""
    
    def __init__(self, kg_file="knowledge_graph.json"):
        self.kg_file = kg_file
        self.nodes = []
        self.edges = []
        self.load_kg()
    
    def load_kg(self):
        """Load knowledge graph from JSON file"""
        try:
            with open(self.kg_file, 'r', encoding='utf-8') as f:
                kg = json.load(f)
            self.nodes = kg.get('nodes', [])
            self.edges = kg.get('edges', [])
            print(f"✅ Loaded {len(self.nodes)} nodes, {len(self.edges)} edges from {self.kg_file}")
        except FileNotFoundError:
            print(f"⚠ Knowledge graph file not found: {self.kg_file}")
            print("   Run `python kg_extractor.py` first to create it.")
            self.nodes = []
            self.edges = []
    
    def query_entities(self, entity_name: str, entity_type: str = None) -> List[Dict]:
        """Query entities by name and optionally type"""
        results = []
        for node in self.nodes:
            if entity_name.lower() in node['label'].lower():
                if entity_type is None or entity_type == node['type']:
                    results.append(node)
        return results
    
    def query_relations(self, entity_name: str, relation_type: str = None) -> List[Dict]:
        """Query relations involving an entity"""
        results = []
        entity_ids = [node['id'] for node in self.nodes 
                     if entity_name.lower() in node['label'].lower()]
        
        for edge in self.edges:
            if edge['source'] in entity_ids or edge['target'] in entity_ids:
                if relation_type is None or relation_type.lower() in edge['label'].lower():
                    # Get full node info
                    source_node = next((n for n in self.nodes if n['id'] == edge['source']), None)
                    target_node = next((n for n in self.nodes if n['id'] == edge['target']), None)
                    
                    if source_node and target_node:
                        results.append({
                            "source": source_node['label'],
                            "target": target_node['label'],
                            "relation": edge['label'],
                            "context": edge.get('sentence', ''),
                            "doc_id": edge.get('doc_id', '')
                        })
        return results
    
    def semantic_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Semantic search across graph nodes and edges"""
        results = []
        query_lower = query.lower()
        
        # Search in nodes
        for node in self.nodes:
            if query_lower in node['label'].lower() or query_lower in node['type'].lower():
                results.append({
                    "type": "entity",
                    "content": node['label'],
                    "entity_type": node['type'],
                    "score": 1.0,
                    "docs": node.get('docs', [])
                })
        
        # Search in edges
        for edge in self.edges:
            if query_lower in edge['label'].lower():
                source_node = next((n for n in self.nodes if n['id'] == edge['source']), None)
                target_node = next((n for n in self.nodes if n['id'] == edge['target']), None)
                
                if source_node and target_node:
                    results.append({
                        "type": "relation",
                        "content": f"{source_node['label']} {edge['label']} {target_node['label']}",
                        "relation": edge['label'],
                        "score": 0.8,
                        "context": edge.get('sentence', '')
                    })
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def get_entity_neighbors(self, entity_name: str, depth: int = 1) -> Dict:
        """Get neighboring entities in the graph"""
        entity_nodes = [n for n in self.nodes if entity_name.lower() in n['label'].lower()]
        
        if not entity_nodes:
            return {"entity": entity_name, "neighbors": []}
        
        neighbors = []
        entity_id = entity_nodes[0]['id']
        
        for edge in self.edges:
            if edge['source'] == entity_id:
                target_node = next((n for n in self.nodes if n['id'] == edge['target']), None)
                if target_node:
                    neighbors.append({
                        "relation": edge['label'],
                        "neighbor": target_node['label'],
                        "direction": "outgoing"
                    })
            elif edge['target'] == entity_id:
                source_node = next((n for n in self.nodes if n['id'] == edge['source']), None)
                if source_node:
                    neighbors.append({
                        "relation": edge['label'],
                        "neighbor": source_node['label'],
                        "direction": "incoming"
                    })
        
        return {
            "entity": entity_nodes[0]['label'],
            "neighbors": neighbors
        }
    
    def add_custom_relation(self, subject: str, predicate: str, object: str, context: str = ""):
        """Add a custom relation to the graph (for testing)"""
        # Find or create subject node
        subject_node = next((n for n in self.nodes if n['label'] == subject), None)
        if not subject_node:
            subject_node = {
                "id": len(self.nodes),
                "label": subject,
                "type": "CUSTOM",
                "docs": ["manual"]
            }
            self.nodes.append(subject_node)
        
        # Find or create object node
        object_node = next((n for n in self.nodes if n['label'] == object), None)
        if not object_node:
            object_node = {
                "id": len(self.nodes),
                "label": object,
                "type": "CUSTOM",
                "docs": ["manual"]
            }
            self.nodes.append(object_node)
        
        # Add edge
        new_edge = {
            "source": subject_node['id'],
            "target": object_node['id'],
            "label": predicate,
            "doc_id": "manual",
            "sentence": context
        }
        self.edges.append(new_edge)
        
        print(f"✅ Added relation: {subject} --[{predicate}]--> {object}")
    
    def save(self, filename: str = None):
        """Save current graph to file"""
        if filename is None:
            filename = self.kg_file
        
        kg = {
            "nodes": self.nodes,
            "edges": self.edges,
            "statistics": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges)
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(kg, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Graph saved to {filename}")

def test_graph_db():
    """Test the graph database functionality"""
    print("="*60)
    print("GRAPH DATABASE TEST")
    print("="*60)
    
    # Initialize
    db = MockGraphDB()
    
    # Query example
    print("\n🔍 Querying entities with 'caste':")
    entities = db.query_entities("caste")
    for e in entities[:3]:
        print(f"   - {e['label']} [{e['type']}]")
    
    print("\n🔍 Semantic search for 'democracy':")
    results = db.semantic_search("democracy", top_k=2)
    for r in results:
        print(f"   - {r['type']}: {r['content']}")
    
    print("\n🔍 Getting neighbors of 'Ambedkar' (if exists):")
    neighbors = db.get_entity_neighbors("Ambedkar")
    if neighbors['neighbors']:
        for n in neighbors['neighbors'][:3]:
            print(f"   - {neighbors['entity']} --[{n['relation']}]--> {n['neighbor']}")
    else:
        print("   No Ambedkar node found in graph (needs manual addition)")
    
    # Add a manual relation for testing
    print("\n➕ Adding custom relation:")
    db.add_custom_relation(
        subject="Ambedkar",
        predicate="advocated_for",
        object="equality",
        context="Ambedkar fought for equality and social justice"
    )
    
    # Save
    db.save("knowledge_graph_enhanced.json")
    
    print("\n✅ Graph database test complete!")

if __name__ == "__main__":
    test_graph_db()