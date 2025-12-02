"""
AmbedkarGPT - AWS Bedrock Version
Uses AWS Bedrock (Claude 3) with fallback to local Ollama
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Import our Bedrock integration
from bedrock_integration import BedrockLLM

# Disable SSL verification for HuggingFace
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'

def load_documents(corpus_path="corpus"):
    """Load all text documents from corpus folder"""
    documents = []
    for i in range(1, 7):
        filename = f"{corpus_path}/speech{i}.txt"
        try:
            loader = TextLoader(filename, encoding="utf-8")
            docs = loader.load()
            documents.extend(docs)
            print(f"✓ Loaded: {filename}")
        except Exception as e:
            print(f"✗ Error loading {filename}: {e}")
    return documents

def create_chunks(documents, chunk_size=500, overlap=50):
    """Split documents into chunks"""
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separator="\n"
    )
    return text_splitter.split_documents(documents)

def create_vector_store(chunks):
    """Create vector store from text chunks"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        print("✓ Vector store created with embeddings")
        return vector_store
    except Exception as e:
        print(f"✗ Embedding failed: {e}")
        print("⚠ Using simple text-based retrieval as fallback")
        return None

def retrieve_documents(question, vector_store, texts=None, top_k=3):
    """Retrieve relevant documents for a question"""
    if vector_store:
        # Use vector similarity search
        docs = vector_store.similarity_search(question, k=top_k)
        return [(doc.page_content, doc.metadata.get('source', 'unknown')) for doc in docs]
    else:
        # Simple text-based fallback
        relevant_texts = []
        for text in texts:
            if any(word.lower() in text.lower() for word in question.lower().split()):
                relevant_texts.append((text, "corpus"))
        
        if not relevant_texts and texts:
            relevant_texts = [(texts[0], "corpus")]
        
        return relevant_texts[:top_k]

def answer_question(question, vector_store, texts=None):
    """Generate answer using Bedrock LLM"""
    # Initialize Bedrock LLM
    llm = BedrockLLM()
    
    # Retrieve context
    retrieved = retrieve_documents(question, vector_store, texts)
    context = "\n\n".join([text for text, _ in retrieved])
    
    # Prepare prompt
    prompt = f"""You are an expert on Dr. B.R. Ambedkar's speeches and writings.
Based ONLY on the following context from his speeches, answer the question.

Context: {context}

Question: {question}

Provide a concise, accurate answer based strictly on the context. If the answer isn't in the context, say "I cannot answer based on the provided documents."

Answer: """
    
    # Generate answer
    try:
        answer = llm.invoke(prompt)
        return answer, retrieved
    except Exception as e:
        return f"Error: {e}", retrieved

def main():
    print("="*60)
    print("AmbedkarGPT - AWS Bedrock Edition")
    print("="*60)
    
    # Load documents
    print("\n📂 Loading documents...")
    documents = load_documents("corpus")
    print(f"   Loaded {len(documents)} documents")
    
    # Create chunks
    print("\n✂ Creating text chunks...")
    chunks = create_chunks(documents, chunk_size=500, overlap=50)
    print(f"   Created {len(chunks)} chunks")
    
    # Create vector store
    print("\n🔍 Creating vector store...")
    vector_store = create_vector_store(chunks)
    
    # Prepare fallback text list
    all_texts = [chunk.page_content for chunk in chunks] if chunks else []
    
    print("\n" + "="*60)
    print("✅ System Ready! Ask questions about Ambedkar's speeches")
    print("   Type 'exit' or 'quit' to end")
    print("="*60)
    
    while True:
        question = input("\n❓ Your question: ").strip()
        
        if question.lower() in ['exit', 'quit', 'bye']:
            print("\n👋 Thank you for using AmbedkarGPT!")
            break
        
        if not question:
            continue
        
        print("🤔 Thinking...")
        answer, sources = answer_question(question, vector_store, all_texts)
        
        print(f"\n✅ Answer: {answer}")
        print(f"\n📚 Sources used: {len(sources)} document(s)")
        for i, (text, source) in enumerate(sources, 1):
            print(f"   {i}. {source} - {text[:100]}...")

if __name__ == "__main__":
    main()