"""
Entity Extractor for Knowledge Graph Construction
Uses spaCy for NER and dependency parsing
"""

import spacy
import json
from typing import List, Dict, Any, Tuple
import re
from collections import defaultdict

class EntityExtractor:
    """Extracts entities and relationships from text using spaCy"""
    
    def __init__(self):
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_sm")
        
        # Entity types to extract (from spaCy NER)
        self.entity_types = {
            "PERSON", "ORG", "GPE", "LOC", "PRODUCT", 
            "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"
        }
        
        # Custom entity patterns for Ambedkar's works
        self.custom_patterns = [
            ("AMBEDKAR", ["ambedkar", "dr. b. r. ambedkar", "b. r. ambedkar"]),
            ("CASTE", ["caste", "varna", "jati", "chaturvarnya"]),
            ("RELIGION", ["hinduism", "buddhism", "islam", "christianity"]),
            ("DOCUMENT", ["constitution", "manusmriti", "vedas", "shastras"]),
            ("CONCEPT", ["untouchability", "dalit", "equality", "liberty", "fraternity"]),
            ("PLACE", ["india", "pakistan", "britain", "columbia university"]),
        ]
    
    def extract_entities(self, text: str, chunk_id: int) -> List[Dict[str, Any]]:
        """Extract entities from text using spaCy NER"""
        doc = self.nlp(text)
        
        entities = []
        seen_entities = set()  # To avoid duplicates
        
        # Extract spaCy entities
        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                entity_text = ent.text.strip()
                entity_key = f"{entity_text}_{ent.label_}"
                
                if entity_key not in seen_entities:
                    entities.append({
                        "text": entity_text,
                        "type": ent.label_,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "chunk_id": chunk_id,
                        "source": "spacy"
                    })
                    seen_entities.add(entity_key)
        
        # Extract custom pattern entities
        text_lower = text.lower()
        for entity_type, patterns in self.custom_patterns:
            for pattern in patterns:
                if pattern in text_lower:
                    # Find all occurrences
                    matches = re.finditer(r'\b' + re.escape(pattern) + r'\b', text_lower)
                    for match in matches:
                        entity_text = text[match.start():match.end()]
                        entity_key = f"{entity_text}_{entity_type}"
                        
                        if entity_key not in seen_entities:
                            entities.append({
                                "text": entity_text,
                                "type": entity_type,
                                "label": entity_type,
                                "start": match.start(),
                                "end": match.end(),
                                "chunk_id": chunk_id,
                                "source": "custom_pattern"
                            })
                            seen_entities.add(entity_key)
        
        # Extract noun phrases as potential entities
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2:  # Multi-word phrases
                entity_text = chunk.text.strip()
                entity_key = f"{entity_text}_NOUN_PHRASE"
                
                if entity_key not in seen_entities and len(entity_text) > 3:
                    entities.append({
                        "text": entity_text,
                        "type": "NOUN_PHRASE",
                        "label": "NOUN_PHRASE",
                        "start": chunk.start_char,
                        "end": chunk.end_char,
                        "chunk_id": chunk_id,
                        "source": "noun_phrase"
                    })
                    seen_entities.add(entity_key)
        
        return entities
    
    def extract_relationships(self, text: str, entities: List[Dict], chunk_id: int) -> List[Dict[str, Any]]:
        """Extract relationships between entities using dependency parsing"""
        doc = self.nlp(text)
        relationships = []
        
        # Create entity position map
        entity_positions = {}
        for entity in entities:
            if entity["start"] < len(text) and entity["end"] <= len(text):
                entity_positions[(entity["start"], entity["end"])] = entity
        
        # Extract subject-verb-object relationships
        for sent in doc.sents:
            sent_entities = []
            for entity in entities:
                if entity["start"] >= sent.start_char and entity["end"] <= sent.end_char:
                    sent_entities.append(entity)
            
            if len(sent_entities) >= 2:
                # Create relationships between entities in same sentence
                for i in range(len(sent_entities)):
                    for j in range(i + 1, len(sent_entities)):
                        rel_type = self._determine_relationship(
                            sent_entities[i], sent_entities[j], sent.text
                        )
                        
                        if rel_type:
                            relationships.append({
                                "source": sent_entities[i]["text"],
                                "target": sent_entities[j]["text"],
                                "type": rel_type,
                                "source_type": sent_entities[i]["type"],
                                "target_type": sent_entities[j]["type"],
                                "chunk_id": chunk_id,
                                "sentence": sent.text[:200]  # Truncate long sentences
                            })
        
        return relationships
    
    def _determine_relationship(self, entity1: Dict, entity2: Dict, sentence: str) -> str:
        """Determine relationship type based on entity types and context"""
        
        # Type-based relationships
        type_pairs = {
            ("PERSON", "ORG"): "MEMBER_OF",
            ("PERSON", "GPE"): "FROM",
            ("PERSON", "EVENT"): "PARTICIPATED_IN",
            ("PERSON", "WORK_OF_ART"): "CREATED",
            ("ORG", "GPE"): "LOCATED_IN",
            ("CONCEPT", "DOCUMENT"): "DISCUSSED_IN",
            ("CASTE", "PERSON"): "AFFECTS",
            ("RELIGION", "PERSON"): "FOLLOWED_BY",
        }
        
        # Check type pairs
        pair = (entity1["type"], entity2["type"])
        reverse_pair = (entity2["type"], entity1["type"])
        
        if pair in type_pairs:
            return type_pairs[pair]
        elif reverse_pair in type_pairs:
            # Return reverse relationship
            return f"RELATED_TO_{type_pairs[reverse_pair]}"
        
        # Check for specific keywords in sentence
        sentence_lower = sentence.lower()
        
        if "wrote" in sentence_lower or "authored" in sentence_lower:
            if entity1["type"] == "PERSON" and entity2["type"] in ["DOCUMENT", "WORK_OF_ART"]:
                return "WROTE"
        
        if "discussed" in sentence_lower or "mentioned" in sentence_lower:
            return "MENTIONS"
        
        if "opposed" in sentence_lower or "against" in sentence_lower:
            return "OPPOSES"
        
        if "supported" in sentence_lower or "for" in sentence_lower:
            return "SUPPORTS"
        
        if "related to" in sentence_lower or "connected to" in sentence_lower:
            return "RELATED_TO"
        
        # Default relationship
        return "ASSOCIATED_WITH"
    
    def process_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process all chunks to extract entities and relationships"""
        print("="*60)
        print("KNOWLEDGE GRAPH: ENTITY EXTRACTION")
        print("="*60)
        
        all_entities = []
        all_relationships = []
        entity_count = 0
        relationship_count = 0
        
        for chunk in chunks:
            chunk_id = chunk.get("id", 0)
            text = chunk.get("text", "")
            
            if text:
                # Extract entities
                entities = self.extract_entities(text, chunk_id)
                all_entities.extend(entities)
                entity_count += len(entities)
                
                # Extract relationships
                if len(entities) >= 2:
                    relationships = self.extract_relationships(text, entities, chunk_id)
                    all_relationships.extend(relationships)
                    relationship_count += len(relationships)
        
        # Consolidate duplicate entities
        consolidated_entities = self._consolidate_entities(all_entities)
        
        print(f"Extracted {len(consolidated_entities)} unique entities")
        print(f"Extracted {len(all_relationships)} relationships")
        print("="*60)
        
        return {
            "entities": consolidated_entities,
            "relationships": all_relationships,
            "stats": {
                "total_entities": len(consolidated_entities),
                "total_relationships": len(all_relationships),
                "entity_types": self._count_entity_types(consolidated_entities)
            }
        }
    
    def _consolidate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Consolidate duplicate entities"""
        entity_map = {}
        
        for entity in entities:
            key = f"{entity['text'].lower()}_{entity['type']}"
            
            if key not in entity_map:
                entity_map[key] = {
                    "id": len(entity_map),
                    "text": entity["text"],
                    "type": entity["type"],
                    "label": entity["label"],
                    "chunk_ids": [entity["chunk_id"]],
                    "occurrences": 1,
                    "sources": [entity["source"]]
                }
            else:
                # Update existing entity
                if entity["chunk_id"] not in entity_map[key]["chunk_ids"]:
                    entity_map[key]["chunk_ids"].append(entity["chunk_id"])
                entity_map[key]["occurrences"] += 1
                if entity["source"] not in entity_map[key]["sources"]:
                    entity_map[key]["sources"].append(entity["source"])
        
        return list(entity_map.values())
    
    def _count_entity_types(self, entities: List[Dict]) -> Dict[str, int]:
        """Count entities by type"""
        type_counts = defaultdict(int)
        for entity in entities:
            type_counts[entity["type"]] += 1
        return dict(type_counts)
    
    def save_extractions(self, extractions: Dict[str, Any], output_path: str):
        """Save extractions to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(extractions, f, indent=2, ensure_ascii=False)
        
        print(f"Entity extractions saved to {output_path}")
        
        # Print statistics
        stats = extractions["stats"]
        print("\nENTITY TYPE DISTRIBUTION:")
        for entity_type, count in stats["entity_types"].items():
            print(f"  {entity_type}: {count}")

def main():
    """Test entity extraction"""
    # Load chunks
    chunks_path = "../../data/processed/chunks.json"
    with open(chunks_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        chunks = data["chunks"]
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Initialize extractor
    extractor = EntityExtractor()
    
    # Process chunks (sample first 20 chunks for testing)
    sample_chunks = chunks[:20]
    extractions = extractor.process_chunks(sample_chunks)
    
    # Save extractions
    extractor.save_extractions(extractions, "../../data/processed/entity_extractions.json")
    
    # Print sample entities
    print("\nSAMPLE ENTITIES:")
    for entity in extractions["entities"][:10]:
        print(f"  {entity['text']} ({entity['type']}) - appears in {entity['occurrences']} chunks")
    
    # Print sample relationships
    print("\nSAMPLE RELATIONSHIPS:")
    for rel in extractions["relationships"][:5]:
        print(f"  {rel['source']} --[{rel['type']}]--> {rel['target']}")

if __name__ == "__main__":
    main()