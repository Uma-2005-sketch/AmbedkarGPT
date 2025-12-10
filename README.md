
## Project Overview

**AmbedkarGPT** is a comprehensive RAG (Retrieval-Augmented Generation) ecosystem for Dr. B.R. Ambedkar's works, featuring multiple implementations:

1. **Enhanced AmbedkarGPT** (Original): Cultural context-aware RAG with AWS Bedrock integration
2. **SEMRAG Implementation** (New): Research-level semantic knowledge-augmented RAG based on SEMRAG paper
3. **94-Page PDF Processing System** (Latest): Complete pipeline for processing Ambedkar's 94-page book with enhanced search capabilities

## Latest Implementation: 94-Page PDF Processing System

### Overview
This latest implementation processes Dr. B.R. Ambedkar's complete 94-page book "Annihilation of Caste" with advanced semantic search and knowledge graph capabilities.

### Key Features
- **Semantic Chunking**: 339 semantic chunks from 94-page PDF
- **Knowledge Graph**: 490 entities with relationships
- **Community Detection**: 73 thematic communities
- **Enhanced Search**: Dual search strategy (local + global)
- **LLM Integration**: Ollama Mistral for answer generation

### Performance Metrics
- **Chunks Processed**: 339 semantic chunks
- **Entities Extracted**: 490 nodes in knowledge graph
- **Communities Detected**: 73 thematic groups
- **Search Results**: 200+ relevant chunks per query
- **Answer Quality**: Context-aware, citation-supported responses

## Project Structure

```
ambedkargpt/
├── data/                            # Data files
│   ├── Ambedkar_book.pdf           # 94-page PDF
│   └── processed/                  # Processed data
│       ├── chunks.json            # 339 semantic chunks
│       ├── knowledge_graph.pkl    # 490-node knowledge graph
│       ├── communities.json       # 73 communities
│       └── entity_extractions.json
├── src/                           # Source code
│   ├── pipeline/                  # Core pipeline
│   │   ├── ambedkargpt.py        # Original pipeline
│   │   ├── ambedkargpt_enhanced.py # Enhanced version
│   │   ├── test_interactive.py   # Interactive tester
│   │   ├── test_simple.py        # Simple tester
│   │   └── test_multiple_questions.py
│   ├── chunking/                  # Semantic chunking
│   ├── graph/                     # KG construction
│   └── retrieval/                 # Search algorithms
├── README.md                      # This file
├── requirements.txt               # Dependencies
└── .gitignore                    # Git ignore rules
```

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/Uma-2005-sketch/AmbedkarGPT.git
cd AmbedkarGPT

# Create virtual environment
python -m venv semrag_env
semrag_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install Ollama for local LLM
ollama pull mistral
```

### Run Latest Implementation
```bash
cd src/pipeline
python test_interactive.py
```

### Example Questions
The system can answer diverse questions including:
- "What is caste according to Ambedkar?"
- "Why did the Jat-Pat-Todak Mandal reject Ambedkar's speech?"
- "How does Ambedkar view Hindu society?"
- "What are the characteristics of caste system?"
- "How does caste differ from class?"

## Technical Implementation

### Core Algorithms
1. **Semantic Chunking**: Cosine similarity-based segmentation
2. **Local Graph Search (Equation 4)**: 
   ```
   Score_local = 0.6 × ChunkSimilarity + 0.4 × EntitySimilarity
   ```
3. **Global Graph Search (Equation 5)**:
   ```
   Score_global = 0.5 × CommunitySimilarity + 0.5 × ChunkSimilarity
   ```
4. **Knowledge Graph Construction**: Entity extraction and relationship mapping
5. **Community Detection**: Louvain algorithm for thematic grouping

### Search Strategies
- **Keyword Fallback**: Jaccard similarity for proper nouns
- **Dynamic Thresholds**: 0.15 for chunks, 0.3 for entities
- **Result Deduplication**: Unique chunk selection
- **Combined Retrieval**: Local + global search integration

## Usage Examples

### Interactive Q&A:
```bash
cd src/pipeline
python test_interactive.py

Enter your question: Why does Ambedkar call Chaturvarnya impractical and harmful?
```

### Programmatic Usage:
```python
from ambedkargpt_enhanced import AmbedkarGPT

# Initialize system
ambedkargpt = AmbedkarGPT()

# Ask questions
question = "What is caste according to Ambedkar?"
results = ambedkargpt.combined_search(question)
answer = ambedkargpt.generate_answer(question, results['combined_results'])
print(answer)
```

### Test Multiple Questions:
```bash
python test_multiple_questions.py
```

## System Features

### Data Processing
- **PDF Processing**: 94-page document segmentation
- **Entity Extraction**: 490 named entities with relationships
- **Graph Construction**: Network building with embeddings
- **Community Detection**: 73 thematic clusters

### Search Capabilities
- **Semantic Search**: Cosine similarity with SentenceTransformers
- **Graph Search**: Entity and community-based retrieval
- **Keyword Search**: Fallback for proper nouns
- **Hybrid Search**: Combined local and global strategies

### LLM Integration
- **Model**: Ollama Mistral with temperature 0.1
- **Prompt Engineering**: Context-specific templates
- **Citation Support**: Chunk references in answers
- **Error Handling**: Graceful degradation

## Project Evolution

### Phase 1: Basic RAG
- Simple Q&A system for Ambedkar's speeches
- Vector similarity retrieval
- Basic evaluation metrics

### Phase 2: Enhanced AmbedkarGPT
- Cultural context preservation
- AWS Bedrock integration
- Knowledge graph enhancement
- Comprehensive evaluation suite

### Phase 3: SEMRAG Implementation
- Research paper implementation
- Semantic chunking (Algorithm 1)
- Dual retrieval system (Equations 4 & 5)
- Community detection

### Phase 4: 94-Page PDF Processing (Current)
- Complete book processing
- Enhanced search capabilities
- Interactive testing interface
- Production-ready pipeline

## Requirements

See `requirements.txt` for complete dependency list.

### Core Dependencies:
```
sentence-transformers==2.2.2
networkx==3.1
numpy==1.24.3
langchain-community==0.0.10
langchain-core==0.1.0
langchain-ollama==0.1.0
scikit-learn==1.3.0
scipy==1.11.3
```

### LLM Requirements:
- **Ollama** with **Mistral** model (local)
- Optional: AWS Bedrock credentials (for original system)

## Testing

### Run Latest Tests:
```bash
cd src/pipeline

# Test basic functionality
python test_simple.py

# Test multiple questions
python test_multiple_questions.py

# Interactive testing
python test_interactive.py
```

## Results

### 94-Page PDF Processing System:
- **Chunks Created**: 339 semantic chunks
- **Knowledge Graph**: 490 nodes, 73 communities
- **Search Performance**: 200+ relevant chunks per query
- **Answer Generation**: Context-aware responses
- **Response Time**: 45-110 seconds depending on query

### Validation:
- Specific factual questions (Jat-Pat-Todak Mandal)
- Conceptual questions (caste definition)
- Comparative questions (caste vs class)
- Analytical questions (Chaturvarnya criticism)

## Future Enhancements

1. **Performance Optimization**
   - Embedding caching
   - Parallel search execution
   - Response time reduction

2. **Feature Additions**
   - Web interface
   - Batch query processing
   - Answer confidence scoring
   - Multi-document support

3. **Model Improvements**
   - Fine-tuned embeddings
   - Multiple LLM support
   - Real-time indexing
   - Feedback-based learning

## Acknowledgments

- Based on Dr. B.R. Ambedkar's "Annihilation of Caste"
- Uses SentenceTransformers for semantic embeddings
- Built with NetworkX for graph operations
- Powered by Ollama's Mistral model

## Contact

**Developer**: KATHERAPALLI UMA SHANKAR REDDY  
**Email**: uma.shankar@btech.christuniversity.in  
**GitHub**: [Uma-2005-sketch](https://github.com/Uma-2005-sketch)

