"""
Implementation of Algorithm 1 from SEMRAG paper: Semantic Chunking via Cosine Similarity
"""
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

class SemanticChunker:
    """
    Semantic chunking based on cosine similarity of sentence embeddings
    Implements Algorithm 1 from SEMRAG paper
    """
    
    def __init__(self, threshold=0.7, buffer_size=2, max_tokens=1024, subchunk_tokens=128):
        """
        Args:
            threshold: Cosine similarity threshold for chunking (τ in paper)
            buffer_size: Number of adjacent sentences for buffer merging (b in paper)
            max_tokens: Maximum tokens per chunk (T_max in paper)
            subchunk_tokens: Overlap tokens for sub-chunks
        """
        self.threshold = threshold
        self.buffer_size = buffer_size
        self.max_tokens = max_tokens
        self.subchunk_tokens = subchunk_tokens
        
        # Simple embedding simulation (we'll replace with real model later)
        self.embedding_dim = 384
        
    def _simple_embed(self, text):
        """Simple deterministic embedding for development"""
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        seed = int(text_hash[:8], 16) % 100000
        np.random.seed(seed)
        emb = np.random.randn(self.embedding_dim)
        return emb / np.linalg.norm(emb)
    
    def _buffer_merge(self, sentences):
        """Merge sentences with buffer for context preservation"""
        merged = []
        for i in range(len(sentences)):
            start = max(0, i - self.buffer_size)
            end = min(len(sentences), i + self.buffer_size + 1)
            merged_sentence = ' '.join(sentences[start:end])
            merged.append(merged_sentence)
        return merged
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _cosine_distance(self, vec1, vec2):
        """Calculate cosine distance (1 - similarity)"""
        return 1 - self._cosine_similarity(vec1, vec2)
    
    def _split_with_overlap(self, chunk):
        """Split chunk with overlap if exceeds token limit (Equation 2)"""
        words = word_tokenize(chunk)
        if len(words) <= self.max_tokens:
            return [chunk]
        
        subchunks = []
        i = 0
        while i < len(words):
            end = min(i + self.max_tokens, len(words))
            subchunk_words = words[i:end]
            subchunks.append(' '.join(subchunk_words))
            
            if end == len(words):
                break
                
            # Move with overlap
            i += (self.max_tokens - self.subchunk_tokens)
        
        return subchunks
    
    def chunk_document(self, text):
        """
        Main chunking algorithm (Algorithm 1 from SEMRAG paper)
        
        Args:
            text: Document text to chunk
            
        Returns:
            List of semantically coherent chunks
        """
        # Step 1: Split into sentences
        sentences = sent_tokenize(text)
        
        # Step 2: Buffer merging
        buffered_sentences = self._buffer_merge(sentences)
        
        # Step 3: Create embeddings
        embeddings = [self._simple_embed(sent) for sent in buffered_sentences]
        
        # Step 4: Semantic chunking using cosine similarity
        chunks = []
        current_chunk = []
        
        for i in range(len(buffered_sentences) - 1):
            current_chunk.append(buffered_sentences[i])
            
            # Calculate cosine distance between current and next sentence
            distance = self._cosine_distance(embeddings[i], embeddings[i+1])
            
            # If distance exceeds threshold, start new chunk
            if distance >= self.threshold:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(chunk_text)
                current_chunk = []
        
        # Add the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        # Add the last sentence if not included
        if len(buffered_sentences) > 0 and (not current_chunk or buffered_sentences[-1] not in current_chunk[-1]):
            chunks.append(buffered_sentences[-1])
        
        # Step 5: Enforce token limits with overlap
        final_chunks = []
        for chunk in chunks:
            subchunks = self._split_with_overlap(chunk)
            final_chunks.extend(subchunks)
        
        return final_chunks
    
    def chunk_pdf(self, pdf_path):
        """Extract text from PDF and chunk it"""
        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            return self.chunk_document(text)
        except ImportError:
            print("pypdf not installed. Using dummy text.")
            # For testing without PDF
            test_text = """
            Dr. B.R. Ambedkar was a jurist, economist, and social reformer.
            He fought against social discrimination of Dalits.
            He was the principal architect of the Indian Constitution.
            His work "Annihilation of Caste" criticizes the caste system.
            Ambedkar believed in liberty, equality, and fraternity.
            He converted to Buddhism in 1956.
            """
            return self.chunk_document(test_text)

# Test function
def test_chunker():
    """Test the semantic chunker"""
    chunker = SemanticChunker(threshold=0.7, buffer_size=2)
    
    # Test with sample text
    test_text = """
    Dr. B.R. Ambedkar was born on 14 April 1891. He was an Indian jurist, economist, 
    and social reformer. He campaigned against social discrimination towards Dalits. 
    He was the principal architect of the Indian Constitution. Ambedkar published 
    "Annihilation of Caste" in 1936. In this work, he criticized the caste system. 
    He argued for the destruction of religious scriptures that support caste.
    """
    
    chunks = chunker.chunk_document(test_text)
    
    print("=" * 60)
    print("SEMANTIC CHUNKING TEST (Algorithm 1 from SEMRAG paper)")
    print("=" * 60)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} ({len(word_tokenize(chunk))} tokens):")
        print("-" * 40)
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
    
    print(f"\nTotal chunks: {len(chunks)}")
    return chunks

if __name__ == "__main__":
    test_chunker()