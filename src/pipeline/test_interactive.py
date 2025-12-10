"""
Interactive Test Script for AmbedkarGPT
Clean version for Windows compatibility
"""

from ambedkargpt_enhanced import AmbedkarGPT
import time

def print_header(text):
    """Print header with border"""
    print("\n" + "="*70)
    print(text)
    print("="*70)

def print_section(text):
    """Print section header"""
    print("\n" + "-"*50)
    print(text)
    print("-"*50)

class InteractiveTester:
    """Interactive testing interface for AmbedkarGPT"""
    
    def __init__(self):
        self.ambedkargpt = None
        self.test_history = []
        
    def initialize_system(self):
        """Initialize AmbedkarGPT system"""
        print_header("INITIALIZING AMBEDKARGPT SYSTEM")
        
        print("\nLoading components:")
        print("1. Semantic chunks from 94-page PDF...")
        print("2. Knowledge graph with entities and relationships...")
        print("3. Community detection and summaries...")
        print("4. Local and global search indices...")
        print("5. LLM (Ollama Mistral)...")
        
        self.ambedkargpt = AmbedkarGPT()
        self.ambedkargpt.initialize()
        
        print("\n✓ System initialized successfully!")
        print_header("AMBEDKARGPT READY")
    
    def display_welcome(self):
        """Display welcome message and instructions"""
        print_header("AMBEDKARGPT INTERACTIVE TESTER")
        
        print("\nSYSTEM CAPABILITIES:")
        print("  • Semantic chunking of 94-page PDF")
        print("  • Knowledge graph with 490+ entities")
        print("  • Local Graph Search (Equation 4)")
        print("  • Global Graph Search (Equation 5)")
        print("  • LLM-powered answer generation")
        
        print("\nSAMPLE QUESTIONS TO TRY:")
        print("  1. What is caste according to Ambedkar?")
        print("  2. How does Ambedkar view Hindu society?")
        print("  3. What are the characteristics of caste system?")
        print("  4. Who are the scholars mentioned by Ambedkar?")
        print("  5. How does caste differ from class?")
        
        print("\nCOMMANDS:")
        print("  • Type your question and press Enter")
        print("  • Type 'history' to see previous questions")
        print("  • Type 'stats' to see system statistics")
        print("  • Type 'quit' to exit")
        print("  • Type 'help' to see this message again")
        print_header("READY FOR QUESTIONS")
    
    def process_question(self, question: str):
        """Process a single question"""
        print(f"\n>>> PROCESSING: {question}")
        
        start_time = time.time()
        
        try:
            # Step 1: Combined search
            print("\n[1/3] Performing combined search...")
            search_results = self.ambedkargpt.combined_search(question)
            
            # Display search results
            metrics = search_results["search_metrics"]
            print(f"   Found {metrics['local_chunks_found']} local chunks")
            print(f"   Found {metrics['global_chunks_found']} global chunks")
            print(f"   Using {metrics['total_unique_chunks']} unique chunks")
            
            # Step 2: Generate answer
            print("[2/3] Generating answer with LLM...")
            answer = self.ambedkargpt.generate_answer(
                question, 
                search_results["combined_results"]
            )
            
            # Step 3: Calculate timing
            end_time = time.time()
            response_time = end_time - start_time
            
            # Store in history
            self.test_history.append({
                "question": question,
                "answer": answer,
                "response_time": response_time,
                "chunks_used": len(search_results["combined_results"]),
                "timestamp": time.strftime("%H:%M:%S")
            })
            
            # Display answer
            print_section("ANSWER")
            print(f"\n{answer}")
            
            # Display context
            print_section("CONTEXT USED (Top 3 chunks)")
            
            for i, chunk in enumerate(search_results["combined_results"][:3], 1):
                print(f"\nChunk {i} (Score: {chunk['score']:.3f}):")
                print(f"   {chunk['text'][:150]}...")
                if "entities" in chunk and chunk["entities"]:
                    print(f"   Related entities: {', '.join(chunk['entities'][:3])}")
            
            # Display metrics
            print_section("PERFORMANCE METRICS")
            print(f"Response time: {response_time:.2f} seconds")
            print(f"Chunks retrieved: {metrics['total_unique_chunks']}")
            print(f"Local search: {metrics['local_chunks_found']} chunks")
            print(f"Global search: {metrics['global_chunks_found']} chunks")
            
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
    
    def show_history(self):
        """Display question history"""
        if not self.test_history:
            print("\nNo questions asked yet.")
            return
        
        print_header("QUESTION HISTORY")
        
        for i, entry in enumerate(self.test_history, 1):
            print(f"\n{i}. {entry['question']}")
            print(f"   Time: {entry['timestamp']} | Duration: {entry['response_time']:.1f}s")
            print(f"   Chunks used: {entry['chunks_used']}")
            print(f"   Answer preview: {entry['answer'][:100]}...")
    
    def show_stats(self):
        """Display system statistics"""
        if not self.ambedkargpt or not self.ambedkargpt.graph:
            print("\nSystem not fully initialized.")
            return
        
        print_header("SYSTEM STATISTICS")
        
        # Graph stats
        if self.ambedkargpt.graph:
            print(f"\nKNOWLEDGE GRAPH:")
            print(f"   Nodes (entities): {self.ambedkargpt.graph.number_of_nodes()}")
            print(f"   Edges (relationships): {self.ambedkargpt.graph.number_of_edges()}")
        
        # Chunk stats
        if self.ambedkargpt.chunks:
            print(f"\nSEMANTIC CHUNKS:")
            print(f"   Total chunks: {len(self.ambedkargpt.chunks)}")
            avg_tokens = sum(len(c.get('text', '').split()) for c in self.ambedkargpt.chunks[:10]) / 10
            print(f"   Avg tokens per chunk: {avg_tokens:.0f}")
        
        # Community stats
        if self.ambedkargpt.communities:
            print(f"\nCOMMUNITIES:")
            print(f"   Total communities: {len(self.ambedkargpt.communities)}")
            largest_comm = max(len(nodes) for nodes in self.ambedkargpt.communities.values())
            print(f"   Largest community: {largest_comm} nodes")
        
        # Test history stats
        if self.test_history:
            print(f"\nTEST HISTORY:")
            print(f"   Questions asked: {len(self.test_history)}")
            avg_time = sum(entry['response_time'] for entry in self.test_history) / len(self.test_history)
            print(f"   Average response time: {avg_time:.2f}s")
            avg_chunks = sum(entry['chunks_used'] for entry in self.test_history) / len(self.test_history)
            print(f"   Average chunks used: {avg_chunks:.1f}")
    
    def run(self):
        """Main interactive loop"""
        self.initialize_system()
        self.display_welcome()
        
        while True:
            try:
                # Get user input
                print("\n>>> Enter your question: ", end="")
                user_input = input().strip()
                
                if not user_input:
                    continue
                
                # Check for commands
                if user_input.lower() == 'quit':
                    print("\nGoodbye! Thank you for testing AmbedkarGPT.")
                    break
                elif user_input.lower() == 'history':
                    self.show_history()
                    continue
                elif user_input.lower() == 'stats':
                    self.show_stats()
                    continue
                elif user_input.lower() == 'help':
                    self.display_welcome()
                    continue
                
                # Process question
                self.process_question(user_input)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                break
            except Exception as e:
                print(f"\nUnexpected error: {e}")
                continue
        
        # Final summary
        if self.test_history:
            print_header("TEST SESSION SUMMARY")
            print(f"\nTotal questions: {len(self.test_history)}")
            print("Session duration: Ready for demo!")
            print("\n✓ AmbedkarGPT testing completed successfully!")

def main():
    """Main function"""
    tester = InteractiveTester()
    tester.run()

if __name__ == "__main__":
    main()