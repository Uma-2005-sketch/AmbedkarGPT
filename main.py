"""
AmbedkarGPT - Final Enhanced Edition
Features:
1. AWS Bedrock integration (with mock fallback)
2. SEMRAG hybrid retrieval (Vector + Knowledge Graph)
3. Contextual embeddings with cultural markers
4. Enhanced evaluation ready
"""

import os
import json
from typing import List, Dict, Tuple

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from bedrock_integration import BedrockLLM
from semrag import SEMRAG
from graph_db import MockGraphDB
from contextual_embedder import ContextualEmbedder

# Disable SSL verification
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'

class EnhancedAmbedkarGPT:
    def __init__(self):
        print("="*60)
        print("AmbedkarGPT - Enhanced Edition")
        print("Features: AWS Bedrock • SEMRAG • Cultural Context • KG")
        print("="*60)
        
        # Initialize components
        self.llm = BedrockLLM()
        self.graph_db = MockGraphDB("knowledge_graph_enhanced.json")
        self.contextual_embedder = ContextualEmbedder()
        self.vector_store = None
        self.semrag = None
        self.all_chunks = []
        
        # Load and process with contextual enhancement
        self._load_with_context()
        
        # Initialize SEMRAG
        self.semrag = SEMRAG(self.vector_store, self.graph_db)
        
        print("\n" + "="*60)
        print("✅ System Ready with Cultural Context Awareness!")
        print("   Type 'exit', 'quit', or 'bye' to end")
        print("="*60)
    
    def _load_with_context(self):
        """Load documents with contextual enhancement"""
        print("\n📂 Loading documents with cultural context...")
        
        # Use contextual embedder to load enhanced chunks
        enhanced_chunks = self.contextual_embedder.load_and_enhance_documents(
            corpus_path="corpus",
            chunk_size=500,
            overlap=50
        )
        
        self.all_chunks = enhanced_chunks
        print(f"   Total enhanced chunks: {len(enhanced_chunks)}")
        
        # Create vector store from enhanced chunks
        print("\n🔍 Creating vector store with contextual embeddings...")
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            self.vector_store = Chroma.from_documents(
                documents=enhanced_chunks,
                embedding=embeddings,
                persist_directory="./chroma_db_enhanced"
            )
            print("   ✓ Vector store created with contextual embeddings")
        except Exception as e:
            print(f"   ⚠ Vector store failed: {e}")
            print("   Using semantic retrieval only")
            self.vector_store = None
    
    def retrieve_context(self, question: str) -> Tuple[str, Dict]:
        """Retrieve context using enhanced SEMRAG"""
        # Enhance query with cultural context
        enhanced_query = self.contextual_embedder.create_contextual_query(question)
        
        if self.semrag:
            # Prepare text list for fallback
            text_list = [chunk.page_content for chunk in self.all_chunks] if not self.vector_store else None
            
            # Hybrid retrieval
            retrieval_result = self.semrag.hybrid_retrieve(
                enhanced_query, 
                vector_texts=text_list,
                top_k=3
            )
            
            # Extract context from combined results
            combined_texts = [text for text, source, score in retrieval_result['combined_results']]
            context = "\n\n".join(combined_texts)
            
            # Show retrieval analysis
            print(f"\n🔍 Retrieval Analysis:")
            print(f"   Vector results: {len(retrieval_result['vector_results'])}")
            print(f"   Graph results: {len(retrieval_result['graph_results'])}")
            print(f"   Text results: {len(retrieval_result['text_results'])}")
            print(f"   Combined: {len(retrieval_result['combined_results'])}")
            
            return context, retrieval_result
        else:
            # Fallback retrieval
            question_lower = question.lower()
            relevant_chunks = []
            for chunk in self.all_chunks:
                if any(word in chunk.page_content.lower() for word in question_lower.split()):
                    relevant_chunks.append(chunk.page_content)
            
            if not relevant_chunks:
                relevant_chunks = [chunk.page_content for chunk in self.all_chunks[:2]]
            
            context = "\n\n".join(relevant_chunks)
            return context, {"combined_results": []}
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer using contextually enhanced prompt"""
        # Create enhanced prompt
        prompt = f"""You are an expert on Dr. B.R. Ambedkar's work with deep understanding of Indian social, cultural, and historical context.

CONTEXTUAL INFORMATION:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Consider the cultural and historical context of the information
2. Address any cultural nuances or sensitivities
3. Provide a comprehensive answer based on the context
4. If information is insufficient, acknowledge limitations
5. Maintain academic rigor while being accessible

ANSWER: """
        
        try:
            answer = self.llm.invoke(prompt)
            return answer
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def analyze_cultural_context(self, question: str, answer: str) -> Dict:
        """Analyze cultural context of Q&A"""
        analysis_prompt = f"""Analyze the cultural and historical context of this Q&A pair:

Question: {question}
Answer: {answer}

Provide analysis in JSON format with:
1. cultural_topics: List of cultural topics addressed
2. historical_period: Relevant historical period
3. sensitivity_score: Score 1-10 for cultural sensitivity
4. context_completeness: Score 1-10 for contextual completeness
"""
        
        try:
            analysis_text = self.llm.invoke(analysis_prompt)
            # Simple extraction (in real system, parse JSON properly)
            return {
                "analysis": analysis_text[:200] + "...",
                "topics": ["caste", "social_justice", "historical_context"]  # Simplified
            }
        except:
            return {"analysis": "Cultural analysis unavailable", "topics": []}
    
    def process_question(self, question: str) -> Dict:
        """Process a question end-to-end"""
        print("\n" + "="*40)
        print(f"QUESTION: {question}")
        print("="*40)
        
        # Step 1: Retrieve context
        context, retrieval_info = self.retrieve_context(question)
        
        # Step 2: Generate answer
        answer = self.generate_answer(question, context)
        
        # Step 3: Cultural analysis
        cultural_analysis = self.analyze_cultural_context(question, answer)
        
        # Prepare result
        result = {
            "question": question,
            "answer": answer,
            "retrieval_info": {
                "total_sources": len(retrieval_info.get('combined_results', [])),
                "vector_sources": len(retrieval_info.get('vector_results', [])),
                "graph_sources": len(retrieval_info.get('graph_results', []))
            },
            "cultural_analysis": cultural_analysis
        }
        
        return result
    
    def display_result(self, result: Dict):
        """Display formatted result"""
        print(f"\n✅ ANSWER: {result['answer']}")
        
        print(f"\n📊 RETRIEVAL METRICS:")
        print(f"   Total sources: {result['retrieval_info']['total_sources']}")
        print(f"   Vector sources: {result['retrieval_info']['vector_sources']}")
        print(f"   Graph sources: {result['retrieval_info']['graph_sources']}")
        
        print(f"\n🎭 CULTURAL ANALYSIS:")
        print(f"   Topics: {', '.join(result['cultural_analysis'].get('topics', []))}")
        print(f"   Analysis: {result['cultural_analysis'].get('analysis', 'N/A')}")
        
        # Show top sources if available
        if hasattr(self, 'last_retrieval_info') and 'combined_results' in self.last_retrieval_info:
            print(f"\n📚 TOP SOURCES:")
            for i, (text, source, score) in enumerate(self.last_retrieval_info['combined_results'][:2], 1):
                source_type = source.split(':')[0]
                preview = text[:80].replace('\n', ' ') + "..."
                print(f"   {i}. [{source_type}] {preview}")
    
    def run(self):
        """Main interaction loop"""
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() in ['exit', 'quit', 'bye']:
                    print("\n👋 Thank you for using Enhanced AmbedkarGPT!")
                    break
                
                if not question:
                    continue
                
                # Process question
                result = self.process_question(question)
                
                # Display result
                self.display_result(result)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session ended by user")
                break
            except Exception as e:
                print(f"\n⚠ Error: {e}")

def main():
    """Main entry point"""
    try:
        print("\n🚀 Initializing Enhanced AmbedkarGPT...")
        chatbot = EnhancedAmbedkarGPT()
        chatbot.run()
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        print("Please ensure all dependencies are installed.")

if __name__ == "__main__":
    main()