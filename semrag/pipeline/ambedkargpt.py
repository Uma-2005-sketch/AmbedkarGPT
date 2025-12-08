"""
Main SEMRAG Pipeline for AmbedkarGPT
Implements complete SEMRAG architecture from the research paper
"""
import os
import json
import time
import yaml
from typing import Dict, List, Any, Optional
import numpy as np


from semrag.chunking.semantic_chunker import SemanticChunker
from semrag.graph.entity_extractor import EntityExtractor
from semrag.graph.graph_builder import KnowledgeGraphBuilder
try:
    from semrag.graph.community_detector import CommunityDetector
except ImportError:
    # Fallback to simple detector
    from semrag.graph.simple_community_detector import SimpleCommunityDetector as CommunityDetector
    print("Note: Using simple community detector (python-louvain not available)")
from semrag.retrieval.local_search import LocalGraphRAGSearch
from semrag.retrieval.global_search import GlobalGraphRAGSearch
from semrag.llm.answer_generator import AnswerGenerator	

class AmbedkarGPT:
    """
    Main SEMRAG pipeline for Ambedkar Q&A system
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the SEMRAG pipeline
        
        Args:
            config_path: Path to configuration YAML file
        """
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                'chunking': {
                    'threshold': 0.7,
                    'buffer_size': 2,
                    'max_tokens': 1024,
                    'subchunk_tokens': 128
                },
                'graph': {
                    'community_method': 'louvain'
                },
                'retrieval': {
                    'local': {
                        'top_k': 5,
                        'tau_e': 0.6,
                        'tau_d': 0.5
                    },
                    'global': {
                        'top_k_communities': 3,
                        'top_k_points': 5
                    }
                },
                'llm': {
                    'model': 'mistral',
                    'temperature': 0.1
                }
            }
        
        # Initialize components
        self.chunker = None
        self.entity_extractor = None
        self.graph_builder = None
        self.community_detector = None
        self.local_searcher = None
        self.global_searcher = None
        self.answer_generator = None
        
        # Data storage
        self.chunks = []
        self.knowledge_graph = None
        self.communities = {}
        self.community_summaries = {}
        self.entity_embeddings = {}
        self.chunk_embeddings = {}
        self.chunk_contents = {}
        
        self.initialized = False
    
    def initialize_components(self):
        """Initialize all pipeline components"""
        print("Initializing SEMRAG pipeline components...")
        
        # Initialize chunker
        self.chunker = SemanticChunker(**self.config['chunking'])
        
        # Initialize graph components
        self.entity_extractor = EntityExtractor()
        self.graph_builder = KnowledgeGraphBuilder()
        self.community_detector = CommunityDetector(
            method=self.config['graph']['community_method']
        )
        
        # Initialize LLM components
        self.answer_generator = AnswerGenerator(**self.config['llm'])
        
        print("✅ All components initialized")
        self.initialized = True
    
    def process_document(self, pdf_path: str) -> Dict:
        """
        Process PDF document through SEMRAG pipeline
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Processing statistics
        """
        if not self.initialized:
            self.initialize_components()
        
        print(f"\n📄 Processing document: {pdf_path}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Semantic Chunking (Algorithm 1)
        print("\n1. Semantic Chunking (Algorithm 1)...")
        self.chunks = self.chunker.chunk_pdf(pdf_path)
        print(f"   Created {len(self.chunks)} chunks")
        
        # Step 2: Extract entities and build knowledge graph
        print("\n2. Building Knowledge Graph...")
        for i, chunk in enumerate(self.chunks):
            # Extract entities
            entities = self.entity_extractor.extract_entities(chunk)
            relations = self.entity_extractor.extract_relations(chunk, entities)
            
            # Add to graph
            self.graph_builder.add_entities(entities, chunk_id=i)
            if relations:
                self.graph_builder.add_relations(relations)
            
            # Store chunk content
            self.chunk_contents[i] = chunk
        
        self.knowledge_graph = self.graph_builder.graph
        graph_stats = self.graph_builder.get_graph_stats()
        print(f"   Graph built: {graph_stats['total_nodes']} nodes, {graph_stats['total_edges']} edges")
        
        # Step 3: Community Detection
        print("\n3. Community Detection...")
        self.communities = self.community_detector.detect_communities(self.knowledge_graph)
        self.community_summaries = self.community_detector.generate_community_summaries(
            self.knowledge_graph, self.communities
        )
        print(f"   Detected {len(set(self.communities.values()))} communities")
        
        # Step 4: Create embeddings for retrieval
        print("\n4. Creating embeddings...")
        self._create_embeddings()
        
        # Step 5: Initialize retrieval systems
        print("\n5. Initializing retrieval systems...")
        self._initialize_retrieval()
        
        elapsed = time.time() - start_time
        
        # Compile statistics
        stats = {
            'processing_time': elapsed,
            'chunks': len(self.chunks),
            'graph_nodes': graph_stats['total_nodes'],
            'graph_edges': graph_stats['total_edges'],
            'communities': len(set(self.communities.values())),
            'entity_embeddings': len(self.entity_embeddings),
            'chunk_embeddings': len(self.chunk_embeddings)
        }
        
        print(f"\n✅ Document processing complete in {elapsed:.2f}s")
        print(f"   - Chunks: {stats['chunks']}")
        print(f"   - Graph: {stats['graph_nodes']} nodes, {stats['graph_edges']} edges")
        print(f"   - Communities: {stats['communities']}")
        
        return stats
    
    def _create_embeddings(self):
        """Create embeddings for entities and chunks"""
        # Simple deterministic embeddings for development
        import hashlib
        
        # Entity embeddings
        for node, data in self.knowledge_graph.nodes(data=True):
            if 'label' in data and data.get('type') != 'chunk':
                entity_id = data['label'].lower().replace(' ', '_')
                text_hash = hashlib.md5(entity_id.encode()).hexdigest()
                seed = int(text_hash[:8], 16) % 100000
                np.random.seed(seed)
                emb = np.random.randn(384)
                self.entity_embeddings[entity_id] = emb / np.linalg.norm(emb)
        
        # Chunk embeddings
        for chunk_id, chunk_text in self.chunk_contents.items():
            text_hash = hashlib.md5(chunk_text.encode()).hexdigest()
            seed = int(text_hash[:8], 16) % 100000
            np.random.seed(seed)
            emb = np.random.randn(384)
            self.chunk_embeddings[chunk_id] = emb / np.linalg.norm(emb)
    
    def _initialize_retrieval(self):
        """Initialize retrieval systems"""
        # Local search (Equation 4)
        self.local_searcher = LocalGraphRAGSearch(
            knowledge_graph=self.knowledge_graph,
            entity_embeddings=self.entity_embeddings,
            chunk_embeddings=self.chunk_embeddings
        )
        
        # Global search (Equation 5)
        self.global_searcher = GlobalGraphRAGSearch(
            communities=self.communities,
            community_summaries=self.community_summaries,
            chunk_contents=self.chunk_contents
        )
    
    def answer_question(self, question: str, method: str = 'hybrid') -> Dict:
        """
        Answer a question using SEMRAG pipeline
        
        Args:
            question: User question
            method: 'local', 'global', or 'hybrid'
            
        Returns:
            Answer with metadata
        """
        if not self.initialized or self.knowledge_graph is None:
            return {
                'error': 'Pipeline not initialized. Please process a document first.',
                'question': question
            }
        
        print(f"\n❓ Question: {question}")
        print(f"🔍 Method: {method}")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            # Perform retrieval based on method
            if method == 'local':
                # Local Graph RAG Search (Equation 4)
                local_results = self.local_searcher.search(
                    question,
                    top_k=self.config['retrieval']['local']['top_k'],
                    tau_e=self.config['retrieval']['local']['tau_e'],
                    tau_d=self.config['retrieval']['local']['tau_d']
                )
                
                # Generate answer
                response = self.answer_generator.generate_from_local_rag(
                    question, local_results
                )
                
            elif method == 'global':
                # Global Graph RAG Search (Equation 5)
                global_results = self.global_searcher.search(
                    question,
                    top_k_communities=self.config['retrieval']['global']['top_k_communities'],
                    top_k_points=self.config['retrieval']['global']['top_k_points']
                )
                
                # Generate answer
                response = self.answer_generator.generate_from_global_rag(
                    question, global_results
                )
                
            else:  # hybrid
                # Both local and global
                local_results = self.local_searcher.search(
                    question,
                    top_k=self.config['retrieval']['local']['top_k'],
                    tau_e=self.config['retrieval']['local']['tau_e'],
                    tau_d=self.config['retrieval']['local']['tau_d']
                )
                
                global_results = self.global_searcher.search(
                    question,
                    top_k_communities=self.config['retrieval']['global']['top_k_communities'],
                    top_k_points=self.config['retrieval']['global']['top_k_points']
                )
                
                # Generate hybrid answer
                response = self.answer_generator.generate_hybrid_answer(
                    question, local_results, global_results
                )
            
            # Add timing information
            response['total_time'] = time.time() - start_time
            
            # Print summary
            print(f"\n✅ Answer generated in {response['total_time']:.2f}s")
            print(f"📊 Retrieval: {response.get('retrieval_method', 'unknown')}")
            print(f"🤖 Model: {response.get('model', 'unknown')}")
            print(f"\n💬 Answer:")
            print("-" * 40)
            print(response['answer'][:500] + "..." if len(response['answer']) > 500 else response['answer'])
            
            return response
            
        except Exception as e:
            error_msg = f"Error answering question: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'error': error_msg,
                'question': question,
                'method': method,
                'total_time': time.time() - start_time
            }
    
    def save_state(self, directory: str):
        """Save pipeline state to directory"""
        os.makedirs(directory, exist_ok=True)
        
        # Save chunks
        with open(os.path.join(directory, 'chunks.json'), 'w') as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        
        # Save graph
        self.graph_builder.save_graph(os.path.join(directory, 'knowledge_graph.json'))
        
        # Save communities
        self.community_detector.save_communities(os.path.join(directory, 'communities.json'))
        
        # Save embeddings (simplified)
        embeddings_data = {
            'entity_embeddings': {k: v.tolist() for k, v in self.entity_embeddings.items()},
            'chunk_embeddings': {k: v.tolist() for k, v in self.chunk_embeddings.items()}
        }
        with open(os.path.join(directory, 'embeddings.json'), 'w') as f:
            json.dump(embeddings_data, f, indent=2)
        
        # Save config
        with open(os.path.join(directory, 'config.yaml'), 'w') as f:
            yaml.dump(self.config, f)
        
        print(f"✅ Pipeline state saved to {directory}")
    
    def load_state(self, directory: str):
        """Load pipeline state from directory"""
        # Load chunks
        with open(os.path.join(directory, 'chunks.json'), 'r') as f:
            self.chunks = json.load(f)
        
        # Load graph
        self.graph_builder.load_graph(os.path.join(directory, 'knowledge_graph.json'))
        self.knowledge_graph = self.graph_builder.graph
        
        # Load communities
        self.community_detector.load_communities(os.path.join(directory, 'communities.json'))
        self.communities = self.community_detector.communities
        self.community_summaries = self.community_detector.community_summaries
        
        # Load embeddings
        with open(os.path.join(directory, 'embeddings.json'), 'r') as f:
            embeddings_data = json.load(f)
            self.entity_embeddings = {k: np.array(v) for k, v in embeddings_data['entity_embeddings'].items()}
            self.chunk_embeddings = {k: np.array(v) for k, v in embeddings_data['chunk_embeddings'].items()}
        
        # Rebuild chunk contents
        self.chunk_contents = {i: chunk for i, chunk in enumerate(self.chunks)}
        
        # Initialize components
        self.initialize_components()
        self._initialize_retrieval()
        
        print(f"✅ Pipeline state loaded from {directory}")
        print(f"   - Chunks: {len(self.chunks)}")
        print(f"   - Graph nodes: {self.knowledge_graph.number_of_nodes()}")
        print(f"   - Communities: {len(set(self.communities.values()))}")

# Demo function
def run_demo():
    """Run a demonstration of the SEMRAG pipeline"""
    print("=" * 70)
    print("SEMRAG PIPELINE DEMO: AmbedkarGPT")
    print("=" * 70)
    print("Implementation of Semantic RAG with Knowledge Graph (SEMRAG Paper)")
    print()
    
    # Create pipeline
    pipeline = AmbedkarGPT()
    pipeline.initialize_components()
    
    # Use test data if PDF not available
    pdf_path = "data/Ambedkar_book.pdf"
    
    if os.path.exists(pdf_path):
        # Process the actual PDF
        stats = pipeline.process_document(pdf_path)
    else:
        print(f"Note: PDF not found at {pdf_path}")
        print("Using built-in test data for demonstration...")
        
        # Create test chunks
        test_chunks = [
            "Dr. B.R. Ambedkar criticized the caste system as hierarchical and divisive.",
            "He advocated for equality and democracy in Indian Constitution.",
            "Ambedkar believed in liberty, equality, and fraternity as fundamental principles.",
            "His work 'Annihilation of Caste' calls for destruction of caste through religious reform.",
            "Gandhi had different views on caste, seeing it as a social division of labor."
        ]
        
        pipeline.chunks = test_chunks
        pipeline.chunk_contents = {i: chunk for i, chunk in enumerate(test_chunks)}
        
        # Build minimal graph for demo
        pipeline.graph_builder = KnowledgeGraphBuilder()
        for i, chunk in enumerate(test_chunks):
            entities = pipeline.entity_extractor.extract_entities(chunk)
            relations = pipeline.entity_extractor.extract_relations(chunk, entities)
            pipeline.graph_builder.add_entities(entities, chunk_id=i)
            if relations:
                pipeline.graph_builder.add_relations(relations)
        
        pipeline.knowledge_graph = pipeline.graph_builder.graph
        
        # Create embeddings
        pipeline._create_embeddings()
        
        # Initialize retrieval
        pipeline._initialize_retrieval()
        
        stats = {
            'chunks': len(test_chunks),
            'graph_nodes': pipeline.knowledge_graph.number_of_nodes(),
            'graph_edges': pipeline.knowledge_graph.number_of_edges()
        }
    
    print(f"\n📊 System ready with:")
    print(f"   • {stats['chunks']} text chunks")
    print(f"   • {stats.get('graph_nodes', 0)} knowledge graph nodes")
    print(f"   • {stats.get('graph_edges', 0)} relationships")
    
    # Demo questions
    demo_questions = [
        "What is caste system according to Ambedkar?",
        "How did Ambedkar view democracy?",
        "What was Ambedkar's contribution to Indian Constitution?"
    ]
    
    print(f"\n🧪 DEMO QUESTIONS:")
    print("-" * 60)
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n{i}. {question}")
        
        # Try different methods
        for method in ['local', 'global', 'hybrid']:
            print(f"\n   Method: {method.upper()}")
            print("   " + "-" * 40)
            
            response = pipeline.answer_question(question, method=method)
            
            if 'error' not in response:
                print(f"   ⏱️  Time: {response.get('total_time', 0):.2f}s")
                print(f"   📝 Answer preview: {response['answer'][:100]}...")
            else:
                print(f"   ❌ Error: {response['error']}")
    
    print(f"\n{'='*70}")
    print("✅ SEMRAG Pipeline Demo Complete!")
    print("=" * 70)
    
    return pipeline

# Command-line interface
def main():
    """Main entry point for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SEMRAG Pipeline for Ambedkar Q&A')
    parser.add_argument('--pdf', type=str, help='Path to PDF document')
    parser.add_argument('--question', type=str, help='Question to answer')
    parser.add_argument('--method', choices=['local', 'global', 'hybrid'], 
                       default='hybrid', help='Retrieval method')
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    parser.add_argument('--config', type=str, help='Configuration file path')
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
        return
    
    # Initialize pipeline
    pipeline = AmbedkarGPT(config_path=args.config)
    
    if args.pdf:
        # Process PDF
        pipeline.process_document(args.pdf)
    
    if args.question:
        # Answer question
        response = pipeline.answer_question(args.question, method=args.method)
        
        if 'error' not in response:
            print(f"\n📋 RESPONSE SUMMARY:")
            print(f"Question: {response['query']}")
            print(f"Method: {response.get('retrieval_method', 'unknown')}")
            print(f"Time: {response.get('total_time', 0):.2f}s")
            print(f"Model: {response.get('model', 'unknown')}")
            print(f"\n📝 ANSWER:")
            print("-" * 40)
            print(response['answer'])
        else:
            print(f"Error: {response['error']}")
    
    if not args.pdf and not args.question and not args.demo:
        print("No action specified. Use --demo for demonstration or provide --pdf and --question.")
        parser.print_help()

if __name__ == "__main__":
    main()