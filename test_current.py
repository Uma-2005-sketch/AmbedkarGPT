import sys
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.llms import Ollama

# Disable SSL verification
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'

def test_rag():
    print("Testing current RAG system...")
    
    # Load documents
    print("Loading documents...")
    loader = TextLoader("speech.txt", encoding="utf-8")
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s)")
    
    # Split text
    print("Splitting text...")
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator="\n"
    )
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} chunks")
    
    all_texts = [doc.page_content for doc in texts]
    
    # Initialize LLM
    print("Initializing LLM...")
    llm = Ollama(model="mistral")
    
    # Test question
    question = "What is the real remedy for caste system?"
    print(f"\nQuestion: {question}")
    
    # Simple retrieval
    relevant_texts = []
    for text in all_texts:
        if any(word.lower() in text.lower() for word in question.lower().split()):
            relevant_texts.append(text)
    
    if not relevant_texts:
        relevant_texts = all_texts[:2]
    
    context = "\n\n".join(relevant_texts)
    
    prompt = f"""Based on the following context from Dr. B.R. Ambedkar's speech, answer the question.

Context: {context}

Question: {question}

Answer: """
    
    print("Generating answer...")
    answer = llm.invoke(prompt)
    print(f"\nAnswer: {answer}")
    print(f"\nUsed {len(relevant_texts)} text chunks")
    
    return True

if __name__ == "__main__":
    try:
        test_rag()
        print("\n✓ Test completed successfully")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")