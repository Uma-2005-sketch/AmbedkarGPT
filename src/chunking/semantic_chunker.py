"""
Semantic Chunking Algorithm 1 from SemRAG Paper
"""

import re
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import tiktoken  # For token counting
import json

class SemanticChunker:
    """Implements Algorithm 1: Semantic Chunking from SemRAG Paper"""
    
    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 buffer_size: int = 3,
                 similarity_threshold: float = 0.5,
                 max_tokens: int = 1024,
                 overlap_tokens: int = 128):
        
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.buffer_size = buffer_size
        self.similarity_threshold = similarity_threshold
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT tokenizer
        
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken"""
        return len(self.tokenizer.encode(text))
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using pypdf"""
        from pypdf import PdfReader
        
        print(f"Extracting text from PDF: {pdf_path}")
        reader = PdfReader(pdf_path)
        text = ""
        
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            print(f"  Processed page {page_num}/{len(reader.pages)}")
        
        print(f"Extracted {len(text)} characters from PDF")
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK"""
        sentences = sent_tokenize(text)
        
        # Clean sentences
        cleaned_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10:  # Remove very short sentences
                cleaned_sentences.append(sent)
        
        print(f"Split into {len(cleaned_sentences)} sentences")
        return cleaned_sentences
    
    def merge_with_buffer(self, sentences: List[str]) -> List[str]:
        """Merge neighboring sentences using buffer (Algorithm 1, Step 3)"""
        merged = []
        buffer = []
        
        for i, sentence in enumerate(sentences):
            buffer.append(sentence)
            
            # If buffer is full or last sentence
            if len(buffer) == self.buffer_size or i == len(sentences) - 1:
                merged_text = " ".join(buffer)
                merged.append(merged_text)
                buffer = []  # Reset buffer
        
        print(f"Merged into {len(merged)} buffer segments")
        return merged
    
    def create_semantic_chunks(self, buffer_segments: List[str]) -> List[Dict[str, Any]]:
        """Create semantic chunks based on cosine similarity (Algorithm 1, Steps 4-5)"""
        
        # Step 4: Create sentence embeddings
        print("Creating sentence embeddings...")
        embeddings = self.embedding_model.encode(buffer_segments)
        
        # Step 5: Group by cosine similarity
        print("Grouping by cosine similarity...")
        chunks = []
        current_chunk = []
        current_embeddings = []
        
        for i, (segment, embedding) in enumerate(zip(buffer_segments, embeddings)):
            if not current_chunk:
                # Start new chunk
                current_chunk.append(segment)
                current_embeddings.append(embedding)
            else:
                # Calculate similarity with current chunk centroid
                chunk_centroid = np.mean(current_embeddings, axis=0)
                similarity = cosine_similarity([embedding], [chunk_centroid])[0][0]
                
                if similarity >= self.similarity_threshold:
                    # Add to current chunk
                    current_chunk.append(segment)
                    current_embeddings.append(embedding)
                else:
                    # Save current chunk and start new one
                    chunk_text = " ".join(current_chunk)
                    if self.count_tokens(chunk_text) <= self.max_tokens:
                        chunks.append({
                            "id": len(chunks),
                            "text": chunk_text,
                            "sentences": current_chunk.copy(),
                            "token_count": self.count_tokens(chunk_text)
                        })
                    else:
                        # Split oversized chunk
                        chunks.extend(self._split_oversized_chunk(current_chunk))
                    
                    current_chunk = [segment]
                    current_embeddings = [embedding]
        
        # Add last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if self.count_tokens(chunk_text) <= self.max_tokens:
                chunks.append({
                    "id": len(chunks),
                    "text": chunk_text,
                    "sentences": current_chunk.copy(),
                    "token_count": self.count_tokens(chunk_text)
                })
            else:
                chunks.extend(self._split_oversized_chunk(current_chunk))
        
        print(f"Created {len(chunks)} semantic chunks")
        return chunks
    
    def _split_oversized_chunk(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """Split chunks that exceed max_tokens using sliding window"""
        sub_chunks = []
        current_text = ""
        
        for sentence in sentences:
            test_text = current_text + " " + sentence if current_text else sentence
            if self.count_tokens(test_text) <= self.max_tokens - self.overlap_tokens:
                current_text = test_text
            else:
                # Save current sub-chunk
                if current_text:
                    sub_chunks.append({
                        "text": current_text,
                        "token_count": self.count_tokens(current_text),
                        "is_subchunk": True
                    })
                
                # Start new sub-chunk with overlap
                current_text = sentence
        
        # Add last sub-chunk
        if current_text:
            sub_chunks.append({
                "text": current_text,
                "token_count": self.count_tokens(current_text),
                "is_subchunk": True
            })
        
        return sub_chunks
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Complete pipeline: PDF → Text → Sentences → Buffer Merge → Semantic Chunks"""
        
        print("="*60)
        print("SEMANTIC CHUNKING PIPELINE (Algorithm 1)")
        print("="*60)
        
        # Step 1: Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        # Step 2: Split into sentences
        sentences = self.split_into_sentences(text)
        
        # Step 3: Merge with buffer
        buffer_segments = self.merge_with_buffer(sentences)
        
        # Steps 4-5: Create semantic chunks
        chunks = self.create_semantic_chunks(buffer_segments)
        
        # Statistics
        total_tokens = sum(chunk["token_count"] for chunk in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0
        
        print("\n" + "="*60)
        print("CHUNKING STATISTICS:")
        print(f"Total chunks: {len(chunks)}")
        print(f"Total tokens: {total_tokens}")
        print(f"Average tokens per chunk: {avg_tokens:.1f}")
        print(f"Max tokens allowed: {self.max_tokens}")
        print("="*60)
        
        return chunks
    
    def save_chunks(self, chunks: List[Dict[str, Any]], output_path: str):
        """Save chunks to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "chunks": chunks,
                "metadata": {
                    "total_chunks": len(chunks),
                    "total_tokens": sum(c["token_count"] for c in chunks),
                    "buffer_size": self.buffer_size,
                    "similarity_threshold": self.similarity_threshold,
                    "max_tokens": self.max_tokens
                }
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Chunks saved to {output_path}")

def main():
    """Test the semantic chunking"""
    chunker = SemanticChunker(
        buffer_size=3,
        similarity_threshold=0.5,
        max_tokens=1024,
        overlap_tokens=128
    )
    
    # Process the PDF
    chunks = chunker.process_pdf("../../data/Ambedkar_book.pdf")
    
    # Save chunks
    chunker.save_chunks(chunks, "../../data/processed/chunks.json")
    
    # Print sample chunks
    print("\nSAMPLE CHUNKS:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1} ({chunk['token_count']} tokens):")
        print(chunk['text'][:200] + "...")

if __name__ == "__main__":
    main()