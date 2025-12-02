"""
Knowledge Graph Extractor for Ambedkar Speeches
Extracts entities and relationships to build a semantic graph.
"""

import spacy
import json
import os
from typing import List, Dict, Tuple
from collections import defaultdict

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("⚠ SpaCy model not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None

class KGExtractor:
    def __init__(self):
        self.entities = defaultdict(list)
        self.relations = []
        
        # Define Ambedkar-specific entity types
        self.entity_types = {
            "PERSON": ["Ambedkar", "Gandhi", "Buddha", "Periyar"],
            "CONCEPT": ["caste", "democracy", "equality", "rights", "constitution"],
            "ORGANIZATION": ["Congress", "British", "Government"],
            "EVENT": ["Independence", "Poona Pact", "Round Table Conference"]
        }
    
    def extract_from_text(self, text: str, doc_id: str) -> Dict:
        """Extract entities and relations from a single document"""
        if not nlp:
            return {"entities": [], "relations": []}
        
        doc = nlp(text)
        
        # Extract entities
        entities = []
        for ent in doc.ents:
            entity_data = {
                "text": ent.text,
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "doc_id": doc_id
            }
            entities.append(entity_data)
            
            # Store for later relation extraction
            self.entities[ent.text].append(entity_data)
        
        # Extract simple subject-verb-object relations
        relations = self._extract_relations(doc, doc_id, entities)
        
        return {
            "doc_id": doc_id,
            "entities": entities,
            "relations": relations
        }
    
    def _extract_relations(self, doc, doc_id: str, entities: List) -> List:
        """Extract simple relations using dependency parsing"""
        relations = []
        
        for token in doc:
            # Look for verb relations
            if token.pos_ == "VERB":
                subject = None
                object_ = None
                
                # Find subject and object
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subject = child.text
                    elif child.dep_ in ["dobj", "pobj", "attr"]:
                        object_ = child.text
                
                if subject and object_:
                    relation = {
                        "subject": subject,
                        "predicate": token.lemma_,
                        "object": object_,
                        "doc_id": doc_id,
                        "sentence": token.sent.text[:100]
                    }
                    relations.append(relation)
                    self.relations.append(relation)
        
        return relations
    
    def extract_from_corpus(self, corpus_path: str = "corpus") -> Dict:
        """Extract knowledge graph from entire corpus"""
        all_extractions = []
        
        for i in range(1, 7):
            filename = f"{corpus_path}/speech{i}.txt"
            if not os.path.exists(filename):
                continue
            
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                print(f"📄 Processing: {filename}")
                extraction = self.extract_from_text(text, f"speech{i}")
                all_extractions.append(extraction)
                
                print(f"   Found {len(extraction['entities'])} entities, {len(extraction['relations'])} relations")
                
            except Exception as e:
                print(f"✗ Error processing {filename}: {e}")
        
        # Build aggregated knowledge graph
        kg = self._build_knowledge_graph(all_extractions)
        return kg
    
    def _build_knowledge_graph(self, extractions: List) -> Dict:
        """Build consolidated knowledge graph from all extractions"""
        kg = {
            "nodes": [],
            "edges": [],
            "statistics": {}
        }
        
        # Collect unique entities
        entity_map = {}
        for extraction in extractions:
            for entity in extraction['entities']:
                key = f"{entity['text']}_{entity['type']}"
                if key not in entity_map:
                    entity_map[key] = {
                        "id": len(kg['nodes']),
                        "label": entity['text'],
                        "type": entity['type'],
                        "docs": [entity['doc_id']]
                    }
                    kg['nodes'].append(entity_map[key])
                else:
                    if entity['doc_id'] not in entity_map[key]['docs']:
                        entity_map[key]['docs'].append(entity['doc_id'])
        
        # Collect unique relations
        relation_set = set()
        for extraction in extractions:
            for relation in extraction['relations']:
                key = f"{relation['subject']}_{relation['predicate']}_{relation['object']}"
                if key not in relation_set:
                    relation_set.add(key)
                    
                    # Find subject and object node IDs
                    subject_key = f"{relation['subject']}_" + next((k.split('_')[1] for k in entity_map.keys() 
                                                                    if k.startswith(relation['subject'] + "_")), "NOUN")
                    object_key = f"{relation['object']}_" + next((k.split('_')[1] for k in entity_map.keys() 
                                                                  if k.startswith(relation['object'] + "_")), "NOUN")
                    
                    if subject_key in entity_map and object_key in entity_map:
                        kg['edges'].append({
                            "source": entity_map[subject_key]['id'],
                            "target": entity_map[object_key]['id'],
                            "label": relation['predicate'],
                            "doc_id": relation['doc_id'],
                            "sentence": relation['sentence']
                        })
        
        kg['statistics'] = {
            "total_nodes": len(kg['nodes']),
            "total_edges": len(kg['edges']),
            "documents_processed": len(extractions)
        }
        
        return kg
    
    def save_kg(self, kg: Dict, filename: str = "knowledge_graph.json"):
        """Save knowledge graph to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(kg, f, indent=2, ensure_ascii=False)
        print(f"✅ Knowledge graph saved to {filename}")
        print(f"📊 Statistics: {kg['statistics']}")
    
    def print_sample(self, kg: Dict, n_samples: int = 5):
        """Print sample nodes and edges"""
        print("\n" + "="*60)
        print("KNOWLEDGE GRAPH SAMPLE")
        print("="*60)
        
        print(f"\n📌 Sample Nodes (showing {min(n_samples, len(kg['nodes']))}):")
        for i, node in enumerate(kg['nodes'][:n_samples]):
            print(f"   {i+1}. {node['label']} [{node['type']}] - Docs: {node['docs']}")
        
        print(f"\n🔗 Sample Edges (showing {min(n_samples, len(kg['edges']))}):")
        for i, edge in enumerate(kg['edges'][:n_samples]):
            source = kg['nodes'][edge['source']]['label']
            target = kg['nodes'][edge['target']]['label']
            print(f"   {i+1}. {source} --[{edge['label']}]--> {target}")
            print(f"      Context: {edge['sentence'][:80]}...")

def main():
    """Main function to extract and save knowledge graph"""
    print("="*60)
    print("KNOWLEDGE GRAPH EXTRACTION - Ambedkar Speeches")
    print("="*60)
    
    extractor = KGExtractor()
    
    # Extract from corpus
    kg = extractor.extract_from_corpus()
    
    # Save to file
    extractor.save_kg(kg, "knowledge_graph.json")
    
    # Print sample
    extractor.print_sample(kg)
    
    print("\n✅ Knowledge graph extraction complete!")

if __name__ == "__main__":
    main()