"""
AmbedkarGPT Integrated Pipeline
Combines Local and Global Graph Search with LLM Answer Generation
"""

import json
import pickle
import numpy as np
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class AmbedkarGPT:
    """Complete AmbedkarGPT pipeline with Local + Global Graph Search"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm = None
        self.graph = None
        self.chunks = None
        self.communities = None
        self.community_summaries = None
        
    def initialize(self):
        """Initialize all components"""
        print("="*60)
        print("INITIALIZING AMBEDKARGPT")
        print("="*60)
        
        # Load resources
        self._load_resources()
        
        # Initialize LLM
        print("\nInitializing LLM (Ollama Mistral)...")
        try:
            self.llm = Ollama(model="mistral", temperature=0.1)
            print("✓ LLM initialized")
        except Exception as e:
            print(f"⚠ LLM initialization failed: {e}")
            print("  Using mock LLM for demonstration")
            self.llm = MockLLM()
        
        print("\n" + "="*60)
        print("AMBEDKARGPT READY")
        print("="*60)
    
    def _load_resources(self):
        """Load all required resources"""
        print("Loading resources...")
        
        # Load graph
        try:
            with open("../../data/processed/knowledge_graph.pkl", 'rb') as f:
                graph_data = pickle.load(f)
                self.graph = graph_data["graph"]
            print(f"✓ Graph loaded: {self.graph.number_of_nodes()} nodes")
        except Exception as e:
            print(f"⚠ Graph loading failed: {e}")
        
        # Load chunks
        try:
            with open("../../data/processed/chunks.json", 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
                self.chunks = chunks_data["chunks"]
            print(f"✓ Chunks loaded: {len(self.chunks)} chunks")
        except Exception as e:
            print(f"⚠ Chunks loading failed: {e}")
        
        # Load communities
        try:
            with open("../../data/processed/communities.json", 'r', encoding='utf-8') as f:
                communities_data = json.load(f)
                self.communities = communities_data["communities"]
                self.community_summaries = communities_data["summaries"]
            print(f"✓ Communities loaded: {len(self.communities)} communities")
        except Exception as e:
            print(f"⚠ Communities loading failed: {e}")
    
    def local_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Perform local graph search"""
        if not self.graph or not self.chunks:
            return []
        
        query_embedding = self.embedding_model.encode([query])[0]
        relevant_chunks = []
        
        # Find relevant entities
        relevant_entities = []
        for node_id in self.graph.nodes():
            node = self.graph.nodes[node_id]
            if "embedding" in node:
                similarity = self._cosine_similarity(query_embedding, node["embedding"])
                if similarity > 0.3:  # Entity threshold
                    relevant_entities.append({
                        "node_id": node_id,
                        "text": node.get("text", ""),
                        "similarity": similarity
                    })
        
        # Sort entities by similarity
        relevant_entities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Find chunks connected to relevant entities
        chunk_scores = {}
        for entity in relevant_entities[:50]:  # Top 50 entities
            node_id = entity["node_id"]
            chunk_ids = self.graph.nodes[node_id].get("chunk_ids", [])
            
            for chunk_id in chunk_ids:
                if chunk_id < len(self.chunks):
                    chunk = self.chunks[chunk_id]
                    chunk_text = chunk.get("text", "")
                    
                    if chunk_text:
                        chunk_embedding = self.embedding_model.encode([chunk_text])[0]
                        chunk_similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                        
                        # Combined score
                        combined_score = 0.6 * chunk_similarity + 0.4 * entity["similarity"]
                        
                        if combined_score > 0.4:  # Document threshold
                            if chunk_id not in chunk_scores or combined_score > chunk_scores[chunk_id]["score"]:
                                chunk_scores[chunk_id] = {
                                    "chunk_id": chunk_id,
                                    "text": chunk_text,
                                    "score": combined_score,
                                    "entities": [entity["text"]]
                                }
                            else:
                                chunk_scores[chunk_id]["entities"].append(entity["text"])
        
        # Convert to list and sort
        chunks_list = list(chunk_scores.values())
        chunks_list.sort(key=lambda x: x["score"], reverse=True)
        
        # Format results
        formatted_chunks = []
        for chunk in chunks_list[:top_k]:
            formatted_chunks.append({
                "id": chunk["chunk_id"],
                "text": chunk["text"][:500] + "..." if len(chunk["text"]) > 500 else chunk["text"],
                "score": chunk["score"],
                "entities": list(set(chunk["entities"]))[:3]  # Unique entities, top 3
            })
        
        return formatted_chunks
    
    def global_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Perform global graph search (simplified version)"""
        if not self.communities or not self.chunks:
            return []
        
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Find relevant communities (simplified)
        community_scores = []
        for comm_id, summary in self.community_summaries.items():
            summary_embedding = self.embedding_model.encode([summary])[0]
            similarity = self._cosine_similarity(query_embedding, summary_embedding)
            
            if similarity > 0.3:  # Lower threshold for global search
                community_scores.append({
                    "community_id": comm_id,
                    "similarity": similarity,
                    "summary": summary
                })
        
        # Sort communities by similarity
        community_scores.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Get chunks from top communities
        relevant_chunks = []
        for community in community_scores[:2]:  # Top 2 communities
            comm_id = int(community["community_id"])
            
            # Get some chunks (simplified - in real implementation, map chunks to communities)
            for i in range(min(5, len(self.chunks))):
                chunk = self.chunks[i]
                chunk_text = chunk.get("text", "")
                
                if chunk_text:
                    chunk_embedding = self.embedding_model.encode([chunk_text])[0]
                    chunk_similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                    
                    # Combined score
                    combined_score = 0.5 * community["similarity"] + 0.5 * chunk_similarity
                    
                    relevant_chunks.append({
                        "chunk_id": i,
                        "text": chunk_text,
                        "score": combined_score,
                        "community": comm_id
                    })
        
        # Sort and format
        relevant_chunks.sort(key=lambda x: x["score"], reverse=True)
        
        formatted_chunks = []
        for chunk in relevant_chunks[:top_k]:
            formatted_chunks.append({
                "id": chunk["chunk_id"],
                "text": chunk["text"][:500] + "..." if len(chunk["text"]) > 500 else chunk["text"],
                "score": chunk["score"],
                "community": chunk["community"]
            })
        
        return formatted_chunks
    
    def combined_search(self, query: str) -> Dict[str, Any]:
        """Combine local and global search results"""
        print(f"\nProcessing query: '{query}'")
        
        # Perform searches
        local_results = self.local_search(query, top_k=2)
        global_results = self.global_search(query, top_k=1)
        
        # Combine results
        all_chunks = []
        chunk_ids = set()
        
        for chunk in local_results + global_results:
            if chunk["id"] not in chunk_ids:
                all_chunks.append(chunk)
                chunk_ids.add(chunk["id"])
        
        # Sort by score
        all_chunks.sort(key=lambda x: x["score"], reverse=True)
        top_chunks = all_chunks[:3]
        
        return {
            "query": query,
            "local_results": local_results,
            "global_results": global_results,
            "combined_results": top_chunks,
            "search_metrics": {
                "local_chunks_found": len(local_results),
                "global_chunks_found": len(global_results),
                "total_unique_chunks": len(all_chunks)
            }
        }
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate answer using LLM"""
        if not self.llm:
            return "LLM not available. Please initialize Ollama with 'ollama pull mistral'"
        
        # Prepare context
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            context_text += f"\n[Context {i} - Score: {chunk['score']:.3f}]:\n{chunk['text']}\n"
        
        # Create prompt
        template = """You are AmbedkarGPT, an expert on Dr. B.R. Ambedkar's works and ideas.
        Use the provided context from Ambedkar's writings to answer the question.
        
        CONTEXT FROM AMBEDKAR'S WRITINGS:
        {context}
        
        QUESTION: {question}
        
        INSTRUCTIONS:
        1. Answer based ONLY on the provided context
        2. If the context doesn't contain the answer, say "Based on the provided context, I cannot answer this question"
        3. Be precise and factual
        4. Cite relevant parts of the context when appropriate
        5. Keep the answer concise but comprehensive
        
        ANSWER:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            {"context": lambda x: context_text, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        try:
            answer = chain.invoke(query)
            return answer
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def query(self, question: str) -> Dict[str, Any]:
        """Complete query pipeline"""
        print("\n" + "="*60)
        print(f"QUERY: {question}")
        print("="*60)
        
        # Step 1: Combined search
        print("\n1. Performing combined search...")
        search_results = self.combined_search(question)
        
        # Step 2: Generate answer
        print("\n2. Generating answer...")
        answer = self.generate_answer(question, search_results["combined_results"])
        
        # Step 3: Prepare response
        response = {
            "question": question,
            "answer": answer,
            "search_results": search_results,
            "context_used": [
                {
                    "chunk_id": chunk["id"],
                    "preview": chunk["text"][:100] + "...",
                    "score": chunk["score"]
                }
                for chunk in search_results["combined_results"]
            ]
        }
        
        return response
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def print_response(self, response: Dict[str, Any]):
        """Print response in readable format"""
        print("\n" + "="*40)
        print("ANSWER:")
        print("="*40)
        print(response["answer"])
        
        print("\n" + "-"*40)
        print("CONTEXT USED:")
        for i, context in enumerate(response["context_used"], 1):
            print(f"\n  Context {i} (Score: {context['score']:.3f}):")
            print(f"  {context['preview']}")
        
        metrics = response["search_results"]["search_metrics"]
        print("\n" + "-"*40)
        print("SEARCH METRICS:")
        print(f"  Local chunks found: {metrics['local_chunks_found']}")
        print(f"  Global chunks found: {metrics['global_chunks_found']}")
        print(f"  Total unique chunks: {metrics['total_unique_chunks']}")
        print("="*40)

class MockLLM:
    """Mock LLM for testing when Ollama is not available"""
    def invoke(self, text: str) -> str:
        return f"[Mock LLM Response] This is a mock response. For real responses, install Ollama and run 'ollama pull mistral'."

def main():
    """Main function to test AmbedkarGPT"""
    # Initialize system
    ambedkargpt = AmbedkarGPT()
    ambedkargpt.initialize()
    
    # Test queries
    test_queries = [
        "What is caste according to Ambedkar?",
        "How does Ambedkar view the Hindu society?",
        "What are the characteristics of caste system?",
        "How does caste differ from class?"
    ]
    
    print("\n" + "="*60)
    print("TESTING AMBEDKARGPT")
    print("="*60)
    
    for query in test_queries:
        response = ambedkargpt.query(query)
        ambedkargpt.print_response(response)
        
        # Pause between queries
        if query != test_queries[-1]:
            input("\nPress Enter for next query...")

if __name__ == "__main__":
    main()