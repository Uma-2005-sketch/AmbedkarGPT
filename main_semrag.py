"""
AmbedkarGPT - SEMRAG Edition
Uses hybrid retrieval: Vector Search + Knowledge Graph (SEMRAG architecture)
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from bedrock_integration import BedrockLLM
from semrag import SEMRAG
from graph_db import MockGraphDB
import os

# Disable SSL verification
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'

class AmbedkarGPT:
    def __init__(self):
        print("="*60)
        print("AmbedkarGPT - SEMRAG Edition")
        print("Hybrid: Vector Search + Knowledge Graph")
        print("="*60)
        
        # Initialize components
        self.llm = BedrockLLM()
        self.graph_db = MockGraphDB("knowledge_graph_enhanced.json")
        self.vector_store = None
        self.semrag = None
        self.all_texts = []
        
        # Load data
        self._load_and_process_documents()
        
        # Initialize SEMRAG
        self.semrag = SEMRAG(self.vector_store, self.graph_db)
        
        print("\n✅ System Ready! Ask questions about Ambedkar's speeches")
        print("   Type 'exit', 'quit', or 'bye' to end")
        print("="*60)
    
    def _load_and_process_documents(self):
        """Load and process corpus documents"""
        print("\n📂 Loading documents...")
        documents = []
        for i in range(1, 7):
            filename = f"corpus/speech{i}.txt"
            try:
                loader = TextLoader(filename, encoding="utf-8")
                docs = loader.load()
                documents.extend(docs)
                print(f"   ✓ {filename}")
            except Exception as e:
                print(f"   ✗ {filename}: {e}")
        
        print(f"\n✂ Creating text chunks (500 chars, 50 overlap)...")
        text_splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separator="\n"
        )
        chunks = text_splitter.split_documents(documents)
        print(f"   Created {len(chunks)} chunks")
        
        # Store all texts for fallback
        self.all_texts = [chunk.page_content for chunk in chunks]
        
        # Create vector store
        print("\n🔍 Creating vector store...")
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            print("   ✓ Vector store created with embeddings")
        except Exception as e:
            print(f"   ⚠ Vector store failed: {e}")
            print("   Using text-based retrieval only")
            self.vector_store = None
    
    def retrieve_context(self, question: str):
        """Retrieve context using SEMRAG hybrid retrieval"""
        if self.semrag:
            retrieval_result = self.semrag.hybrid_retrieve(
                question, 
                vector_texts=self.all_texts if not self.vector_store else None,
                top_k=3
            )
            
            # Use combined results
            combined_texts = [text for text, source, score in retrieval_result['combined_results']]
            context = "\n\n".join(combined_texts)
            
            # Show retrieval sources
            print(f"\n🔍 Retrieval Analysis:")
            print(f"   Vector results: {len(retrieval_result['vector_results'])}")
            print(f"   Graph results: {len(retrieval_result['graph_results'])}")
            print(f"   Combined: {len(retrieval_result['combined_results'])}")
            
            return context, retrieval_result
        else:
            # Fallback to simple retrieval
            relevant_texts = []
            question_lower = question.lower()
            for text in self.all_texts:
                if any(word in text.lower() for word in question_lower.split()):
                    relevant_texts.append(text)
            
            if not relevant_texts:
                relevant_texts = self.all_texts[:2]
            
            context = "\n\n".join(relevant_texts)
            return context, {"vector_results": [], "graph_results": [], "combined_results": []}
    
    def answer_question(self, question: str):
        """Generate answer using SEMRAG-enhanced prompt"""
        print("\n🤔 Processing...")
        
        # Retrieve context
        context, retrieval_info = self.retrieve_context(question)
        
        # Generate enhanced prompt
        if self.semrag:
            prompt = self.semrag.generate_enhanced_prompt(question, context)
        else:
            prompt = f"""Based on the following context from Dr. B.R. Ambedkar's speeches, answer the question.

Context: {context}

Question: {question}

Answer: """
        
        # Generate answer
        try:
            answer = self.llm.invoke(prompt)
            return answer, retrieval_info
        except Exception as e:
            return f"Error: {e}", retrieval_info
    
    def run(self):
        """Main interaction loop"""
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() in ['exit', 'quit', 'bye']:
                    print("\n👋 Thank you for using AmbedkarGPT SEMRAG Edition!")
                    break
                
                if not question:
                    continue
                
                # Get answer
                answer, retrieval_info = self.answer_question(question)
                
                # Display answer
                print(f"\n✅ Answer: {answer}")
                
                # Display sources if available
                if retrieval_info.get('combined_results'):
                    print(f"\n📚 Sources (top {min(3, len(retrieval_info['combined_results']))}):")
                    for i, (text, source, score) in enumerate(retrieval_info['combined_results'][:3], 1):
                        source_type = source.split(':')[0]
                        preview = text[:80].replace('\n', ' ') + "..."
                        print(f"   {i}. [{source_type}] {preview}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Session ended by user")
                break
            except Exception as e:
                print(f"\n⚠ Error: {e}")

def main():
    """Main entry point"""
    try:
        chatbot = AmbedkarGPT()
        chatbot.run()
    except Exception as e:
        print(f"Failed to initialize system: {e}")
        print("Please ensure all dependencies are installed and corpus files exist.")

if __name__ == "__main__":
    main()