"""
Contextual Embedder with Cultural Markers
Enhances text chunks with metadata before embedding.
"""
import os 
import json
from typing import List, Dict
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

class ContextualEmbedder:
    def __init__(self, metadata_file="corpus_metadata.json"):
        self.metadata = self._load_metadata(metadata_file)
    
    def _load_metadata(self, metadata_file: str) -> Dict:
        """Load speech metadata"""
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠ Metadata file {metadata_file} not found")
            return {}
    
    def _get_speech_metadata(self, filename: str) -> Dict:
        """Get metadata for a speech file"""
        # Extract base filename
        if '/' in filename:
            base_name = filename.split('/')[-1]
        else:
            base_name = filename
        
        return self.metadata.get(base_name, {
            "title": base_name,
            "year": "Unknown",
            "context": "Unknown",
            "cultural_markers": [],
            "key_themes": []
        })
    
    def enhance_chunk(self, chunk_text: str, source_file: str, chunk_index: int) -> Document:
        """Enhance a text chunk with metadata"""
        metadata = self._get_speech_metadata(source_file)
        
        # Create enhanced text with context markers
        enhanced_text = f"""
[CULTURAL CONTEXT]
Title: {metadata['title']}
Year: {metadata['year']}
Context: {metadata['context']}
Cultural Markers: {', '.join(metadata['cultural_markers'])}
Key Themes: {', '.join(metadata['key_themes'])}

[SPEECH CONTENT]
{chunk_text}
"""
        
        # Create document with enhanced metadata
        doc_metadata = {
            "source": source_file,
            "title": metadata['title'],
            "year": metadata['year'],
            "cultural_markers": metadata['cultural_markers'],
            "key_themes": metadata['key_themes'],
            "chunk_index": chunk_index,
            "context_summary": metadata['context']
        }
        
        return Document(
            page_content=enhanced_text.strip(),
            metadata=doc_metadata
        )
    
    def load_and_enhance_documents(self, corpus_path: str = "corpus", 
                                   chunk_size: int = 500, overlap: int = 50) -> List[Document]:
        """Load documents, split into chunks, and enhance with context"""
        enhanced_chunks = []
        
        for i in range(1, 7):
            filename = f"{corpus_path}/speech{i}.txt"
            if not os.path.exists(filename):
                continue
            
            try:
                # Load document
                loader = TextLoader(filename, encoding="utf-8")
                documents = loader.load()
                
                # Split into chunks
                text_splitter = CharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                    separator="\n"
                )
                chunks = text_splitter.split_documents(documents)
                
                # Enhance each chunk
                for idx, chunk in enumerate(chunks):
                    enhanced_chunk = self.enhance_chunk(
                        chunk_text=chunk.page_content,
                        source_file=filename,
                        chunk_index=idx
                    )
                    enhanced_chunks.append(enhanced_chunk)
                
                print(f"✓ Enhanced {len(chunks)} chunks from {filename}")
                
            except Exception as e:
                print(f"✗ Error processing {filename}: {e}")
        
        print(f"\n✅ Total enhanced chunks: {len(enhanced_chunks)}")
        return enhanced_chunks
    
    def create_contextual_query(self, query: str, cultural_focus: str = None) -> str:
        """Enhance query with cultural context"""
        base_query = query
        
        if cultural_focus:
            enhanced_query = f"""
[CULTURAL FOCUS: {cultural_focus.upper()}]
{query}

Please consider historical, social, and cultural context when answering.
"""
        else:
            # Auto-detect cultural focus
            cultural_keywords = {
                "caste": "caste system and social hierarchy",
                "buddha": "Buddhist philosophy and ethics",
                "untouchable": "social exclusion and discrimination",
                "democracy": "political systems and rights",
                "constitution": "legal frameworks and social justice"
            }
            
            focus = None
            query_lower = query.lower()
            for keyword, context in cultural_keywords.items():
                if keyword in query_lower:
                    focus = context
                    break
            
            if focus:
                enhanced_query = f"""
[CULTURAL CONTEXT: {focus}]
{query}

Consider the historical and social implications in your response.
"""
            else:
                enhanced_query = f"""
{query}

Please provide contextually aware response considering historical and cultural background.
"""
        
        return enhanced_query

def test_contextual_embedder():
    """Test the contextual embedder"""
    print("="*60)
    print("CONTEXTUAL EMBEDDER TEST")
    print("="*60)
    
    embedder = ContextualEmbedder()
    
    # Test query enhancement
    test_queries = [
        "What is caste system?",
        "Who was Buddha?",
        "What are untouchables?"
    ]
    
    for query in test_queries:
        enhanced = embedder.create_contextual_query(query)
        print(f"\nQuery: {query}")
        print(f"Enhanced: {enhanced[:150]}...")
    
    print("\n✅ Contextual embedder test complete!")

if __name__ == "__main__":
    import os
    test_contextual_embedder()