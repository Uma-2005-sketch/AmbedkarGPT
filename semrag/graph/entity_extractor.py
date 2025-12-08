"""
Entity extraction for knowledge graph construction
Simplified version without spaCy
"""
import re
from collections import defaultdict

class EntityExtractor:
    """
    Simplified entity extractor for Ambedkar's text
    """
    
    # Predefined entities relevant to Ambedkar's work
    KEY_ENTITIES = {
        'person': ['Ambedkar', 'Gandhi', 'Buddha', 'Nehru', 'Dalits', 'Hindus'],
        'concept': ['caste', 'democracy', 'equality', 'liberty', 'fraternity', 
                   'constitution', 'religion', 'social reform', 'discrimination'],
        'work': ['Annihilation of Caste', 'Indian Constitution', 'Castes in India'],
        'organization': ['Jat-Pat-Todak Mandal', 'Indian National Congress'],
    }
    
    def __init__(self):
        self.entity_patterns = self._build_patterns()
        
    def _build_patterns(self):
        """Build regex patterns for entity extraction"""
        patterns = {}
        
        # Person patterns
        person_keywords = '|'.join(self.KEY_ENTITIES['person'])
        patterns['person'] = re.compile(rf'\b({person_keywords})\b', re.IGNORECASE)
        
        # Concept patterns
        concept_keywords = '|'.join(self.KEY_ENTITIES['concept'])
        patterns['concept'] = re.compile(rf'\b({concept_keywords})\b', re.IGNORECASE)
        
        # Work patterns (handle multi-word)
        for work in self.KEY_ENTITIES['work']:
            patterns[f'work_{work.lower().replace(" ", "_")}'] = re.compile(
                re.escape(work), re.IGNORECASE
            )
            
        return patterns
    
    def extract_entities(self, text):
        """Extract entities from text"""
        entities = []
        
        # Extract persons
        persons = self.entity_patterns['person'].findall(text)
        for person in persons:
            entities.append({
                'text': person,
                'label': 'PERSON',
                'type': 'person'
            })
        
        # Extract concepts
        concepts = self.entity_patterns['concept'].findall(text)
        for concept in concepts:
            entities.append({
                'text': concept,
                'label': 'CONCEPT',
                'type': 'concept'
            })
        
        # Extract works
        for work in self.KEY_ENTITIES['work']:
            if work.lower() in text.lower():
                entities.append({
                    'text': work,
                    'label': 'WORK',
                    'type': 'work'
                })
        
        return entities
    
    def extract_relations(self, text, entities):
        """Extract simple relations between entities"""
        relations = []
        words = text.lower().split()
        
        # Simple relation patterns
        relation_verbs = ['argued', 'criticized', 'supported', 'opposed', 'wrote', 
                         'believed', 'advocated', 'rejected', 'accepted']
        
        for i, word in enumerate(words):
            if word in relation_verbs and i > 0 and i < len(words) - 1:
                # Look for entities before and after the verb
                context_start = max(0, i - 3)
                context_end = min(len(words), i + 4)
                context = ' '.join(words[context_start:context_end])
                
                # Check if context contains entities
                for entity1 in entities:
                    if entity1['text'].lower() in context:
                        for entity2 in entities:
                            if entity2['text'].lower() in context and entity1 != entity2:
                                relations.append({
                                    'subject': entity1['text'],
                                    'relation': word,
                                    'object': entity2['text'],
                                    'context': context
                                })
        
        return relations

# Test function
def test_entity_extractor():
    """Test the entity extractor"""
    extractor = EntityExtractor()
    
    test_text = """
    Ambedkar criticized the caste system in his work Annihilation of Caste.
    He argued for equality and democracy. Gandhi had different views on caste.
    """
    
    print("ENTITY EXTRACTION TEST")
    print("=" * 50)
    
    entities = extractor.extract_entities(test_text)
    print(f"\nExtracted {len(entities)} entities:")
    for i, entity in enumerate(entities, 1):
        print(f"{i}. {entity['text']} ({entity['label']})")
    
    relations = extractor.extract_relations(test_text, entities)
    print(f"\nExtracted {len(relations)} relations:")
    for i, rel in enumerate(relations, 1):
        print(f"{i}. {rel['subject']} -> {rel['relation']} -> {rel['object']}")
    
    return entities, relations

if __name__ == "__main__":
    test_entity_extractor()