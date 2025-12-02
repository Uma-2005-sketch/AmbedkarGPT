from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
import os

# Disable SSL verification
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'

def main():
    print("🚀 Starting AmbedkarGPT - Assignment 1")
    
    # Step 1: Load the document
    print("📖 Loading documents...")
    loader = TextLoader("speech.txt", encoding="utf-8")
    documents = loader.load()
    
    # Step 2: Split text into chunks
    print("✂️ Splitting text into chunks...")
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator="\n"
    )
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} text chunks")
    
    # Step 3: Use a simple embedding workaround
    print("🔮 Using simple embeddings workaround...")
    
    # For now, let's skip embeddings and use simple text matching
    # We'll create a simple vector store with dummy embeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
    except:
        print("❌ Embeddings failed, using simple text-based retrieval...")
        # Simple text-based retrieval as fallback
        vector_store = None
        all_texts = [doc.page_content for doc in texts]
    
    # Step 4: Initialize LLM
    print("🤖 Initializing Mistral LLM...")
    llm = Ollama(model="mistral")
    
    print("🎯 AmbedkarGPT is ready! Ask your questions.")
    print("Type 'exit' to quit")
    
    while True:
        question = input("\n❓ Your question: ").strip()
        
        if question.lower() in ['exit', 'quit']:
            print("👋 Goodbye!")
            break
            
        if not question:
            continue
            
        print("💭 Thinking...")
        try:
            # Simple text-based retrieval
            relevant_texts = []
            for text in all_texts:
                if any(word.lower() in text.lower() for word in question.lower().split()):
                    relevant_texts.append(text)
            
            if not relevant_texts:
                relevant_texts = all_texts[:2]  # Fallback to first chunks
            
            context = "\n\n".join(relevant_texts)
            
            prompt = f"""Based on the following context from Dr. B.R. Ambedkar's speech, answer the question.

Context: {context}

Question: {question}

Answer: """
            
            answer = llm.invoke(prompt)
            print(f"\n📝 Answer: {answer}")
            print(f"\n📚 Sources: {len(relevant_texts)} text chunks used")
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()