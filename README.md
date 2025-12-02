
# AmbedkarGPT 

A comprehensive RAG (Retrieval-Augmented Generation) system for Dr. B.R. Ambedkar's speeches with AWS Bedrock integration, SEMRAG architecture, cultural context preservation, and knowledge graph capabilities.

##  Enhanced Features

### **Core Architecture**
1. **AWS Bedrock Integration** - Claude 3 compatible architecture with local fallback (Ollama/Mistral)
2. **SEMRAG Architecture** - Hybrid Vector + Knowledge Graph retrieval system
3. **Cultural Context Preservation** - Metadata markers and context-aware embeddings
4. **Knowledge Graph Integration** - Entity-relation extraction and semantic querying

### **Evaluation & Analysis**
5. **Cultural Benchmark Suite** - 10-question test suite with 5 evaluation metrics
6. **Contextual Compression Analysis** - Information loss measurement across processing strategies

## 📊 Target Outcomes Achieved

| Outcome | Status | Description |
|---------|--------|-------------|
| 1. AWS RAG v. LLM | ✅ **Implemented** | Bedrock integration with mock/real architecture |
| 2. RAG + Knowledge Graph | ✅ **Implemented** | SEMRAG hybrid retrieval system |
| 3. KG + Contextual Markers | ✅ **Implemented** | Cultural metadata and context-aware embeddings |
| 4. Historical Corpus & Evaluation | ✅ **Enhanced** | Extended evaluation with cultural metrics |
| 5. Cultural Context Benchmark | ✅ **Implemented** | 10-question benchmark with scoring |
| 6. Contextual Compression Analysis | ✅ **Implemented** | Info loss measurement and strategy comparison |

## Project Structure

```
AmbedkarGPT/
├── corpus/                          # Ambedkar's speeches (6 documents)
├── corpus_metadata.json             # Cultural metadata for speeches
├── knowledge_graph.json             # Extracted knowledge graph
├── knowledge_graph_enhanced.json    # Enhanced graph with manual relations
├── cultural_benchmark.json          # Cultural understanding test suite
├── cultural_benchmark_results_*.json # Benchmark evaluation results
├── compression_analysis_*.json      # Compression analysis results
│
├── main.py                          # Enhanced SEMRAG system (entry point)
├── bedrock_integration.py           # AWS Bedrock interface
├── mock_bedrock.py                  # Mock Bedrock client (Ollama fallback)
├── semrag.py                        # SEMRAG hybrid retrieval engine
├── kg_extractor.py                  # Knowledge graph extractor
├── graph_db.py                      # Graph database interface
├── contextual_embedder.py           # Cultural context embedding system
├── cultural_evaluator.py            # Cultural evaluation metrics
├── run_cultural_benchmark.py        # Benchmark runner
├── compression_analyzer.py          # Compression analysis tool
│
├── main_original.py                 # Original RAG system (backup)
├── main_bedrock.py                  # Bedrock-only version
├── main_semrag.py                   # SEMRAG-only version
├── evaluation.py                    # Original evaluation framework
│
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Installation & Setup

### **Prerequisites**
- Python 3.8+
- Ollama (for local LLM fallback)
- AWS Account (optional, for Bedrock)

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Setup Ollama (Local Fallback)**
```bash
ollama pull mistral
```

### **3. Run Enhanced System**
```bash
python main.py
```

## 📈 Performance Metrics

### **Cultural Benchmark Results**
- **Average Cultural Score:** 0.333 (Basic Understanding)
- **Cultural Preservation:** 50% average
- **Best Strategy:** Summarization (0.259 info loss)

### **Retrieval Performance**
- **Hit Rate:** 88% (from original evaluation)
- **Hybrid Retrieval:** Vector + Graph integration active

##  Usage Examples

### **1. Interactive Q&A**
```bash
python main.py
```
Ask questions like:
- "What is the cultural significance of caste system?"
- "How did Ambedkar view democracy?"

### **2. Run Cultural Benchmark**
```bash
python run_cultural_benchmark.py 3
```

### **3. Compression Analysis**
```bash
python compression_analyzer.py 2
```

##  Knowledge Graph Features

The system extracts entities and relationships from speeches:
- **22 nodes** (entities like "Buddha", "Caste", "Democracy")
- **Relations** extracted via semantic parsing
- **Graph queries** for semantic relationships

##  Cultural Context System

- **Metadata Markers:** Year, context, cultural themes for each speech
- **Context-Aware Embeddings:** Enhanced with cultural metadata
- **Cultural Sensitivity Scoring:** Built-in evaluation of responses

##  Advanced Features

### **SEMRAG (Semantic RAG with Graph)**
- Combines vector similarity search with graph semantic search
- Hybrid ranking of results from multiple sources
- Contextual enrichment from knowledge graph

### **AWS Bedrock Integration**
- Ready for Claude 3, Titan, Llama 2
- Mock client for local development
- Easy migration to real AWS credentials

### **Compression Analysis**
- Measures information loss in cultural content processing
- Compares summarization, extraction, paraphrasing strategies
- Recommends optimal processing methods

##  Evaluation Framework

### **Original Metrics**
- Hit Rate, MRR, Precision@K
- ROUGE-L, BLEU, Cosine Similarity
- Answer Relevance, Faithfulness

### **Enhanced Cultural Metrics**
- Cultural Relevance (0-1)
- Contextual Accuracy (0-1)
- Nuance Understanding (0-1)
- Sensitivity Score (0-1)
- Completeness Score (0-1)

##  Migration to Production

### **1. AWS Bedrock Setup**
Replace dummy credentials in `bedrock_integration.py` with real AWS credentials.

### **2. Knowledge Graph Enhancement**
Use Neo4j or AWS Neptune instead of mock graph database.

### **3. Vector Store Fix**
Fix metadata format issue for ChromaDB persistence.

##  Acknowledgments

- Dr. B.R. Ambedkar's speeches as primary corpus
- AWS Bedrock for LLM infrastructure
- LangChain for RAG framework
- Community contributors

