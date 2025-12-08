```markdown

\# AmbedkarGPT: Complete RAG Ecosystem



\## Project Overview

\*\*AmbedkarGPT\*\* is a comprehensive RAG (Retrieval-Augmented Generation) ecosystem for Dr. B.R. Ambedkar's works, featuring two distinct implementations:



1\. \*\*Enhanced AmbedkarGPT\*\* (Original): Cultural context-aware RAG with AWS Bedrock integration

2\. \*\*SEMRAG Implementation\*\* (New): Research-level semantic knowledge-augmented RAG based on SEMRAG paper



\## Project Structure

```

AmbedkarGPT/

├── corpus/                          # Original 6 speeches

├── data/                            # 94-page PDF (SEMRAG)

├── semrag/                          # SEMRAG implementation

│   ├── chunking/                   # Algorithm 1: Semantic chunking

│   ├── graph/                      # KG construction \& community detection

│   ├── retrieval/                  # Equations 4 \& 5: Dual retrieval

│   ├── llm/                        # LLM integration

│   └── pipeline/                   # Main SEMRAG pipeline

├── main.py                         # Original enhanced system

├── demo.py                         # SEMRAG demonstration

├── README.md                       # This file

└── requirements.txt                # Dependencies

```



\## Quick Start



\### Installation

```bash

\# Clone repository

git clone https://github.com/Uma-2005-sketch/AmbedkarGPT

cd AmbedkarGPT



\# Install dependencies

pip install -r requirements.txt



\# Install Ollama for local LLM

ollama pull mistral

```



\### Run Original AmbedkarGPT

```bash

python main.py

```



\### Run SEMRAG Implementation

```bash

python demo.py

```



\## Two Implementations



\### 1. Enhanced AmbedkarGPT (Original)

\*\*Features:\*\*

\- Cultural context preservation with metadata

\- AWS Bedrock integration (with Ollama fallback)

\- Knowledge graph with entity extraction

\- Cultural benchmark evaluation suite

\- Contextual compression analysis



\*\*Usage:\*\*

```python

python main.py

\# Interactive Q\&A with cultural context

```



\### 2. SEMRAG Implementation (Research Paper)

\*\*Features:\*\*

\- \*\*Algorithm 1\*\*: Semantic chunking via cosine similarity

\- \*\*Equation 4\*\*: Local Graph RAG Search

\- \*\*Equation 5\*\*: Global Graph RAG Search

\- Louvain community detection

\- Complete SEMRAG pipeline from research paper



\*\*Usage:\*\*

```python

python demo.py

\# Complete SEMRAG architecture demonstration

```



\## Technical Comparison



| Feature | Enhanced AmbedkarGPT | SEMRAG Implementation |

|---------|---------------------|----------------------|

| \*\*Architecture\*\* | Custom RAG + Cultural Context | \*\*SEMRAG Paper Exact\*\* |

| \*\*Chunking\*\* | Fixed-size (500 chars) | \*\*Algorithm 1\*\* Semantic |

| \*\*Knowledge Graph\*\* | Basic NER (1 edge) | Full + \*\*Community Detection\*\* |

| \*\*Retrieval\*\* | Vector similarity | \*\*Dual Retrieval\*\* (Eq 4 \& 5) |

| \*\*Evaluation\*\* | Cultural benchmark suite | Research paper compliance |

| \*\*LLM\*\* | AWS Bedrock + Ollama | Ollama Mistral |



\## SEMRAG Architecture Details



\### Core Algorithms Implemented:

1\. \*\*Algorithm 1\*\*: Semantic chunking via cosine similarity

&nbsp;  ```python

&nbsp;  # Implements: g = {c\_i | d(c\_i, c\_{i+k}) < τ}

&nbsp;  ```



2\. \*\*Equation 4\*\*: Local Graph RAG Search

&nbsp;  ```python

&nbsp;  # Implements: D\_retrieved = Top\_k({v ∈ V, g ∈ G | sim(v, Q+H) > τ\_e ∧ sim(g, v) > τ\_d})

&nbsp;  ```



3\. \*\*Equation 5\*\*: Global Graph RAG Search

&nbsp;  ```python

&nbsp;  # Implements: D\_retrieved = Top\_k(∪{r ∈ R\_Top-K(Q)} ∪{c\_i ∈ C\_r} (∪{p\_j ∈ c\_i} (p\_j, score(p\_j, Q))))

&nbsp;  ```



\### Performance Metrics (SEMRAG):

\- \*\*Document Processing\*\*: 1.29s for 94-page PDF

\- \*\*Semantic Chunks\*\*: 2136 chunks created

\- \*\*Knowledge Graph\*\*: 1720 nodes, 3175 edges

\- \*\*Communities\*\*: 9 thematic groups detected

\- \*\*Answer Generation\*\*: ~92s with Mistral LLM



\## Usage Examples



\### Interactive Q\&A (Original):

```bash

python main.py

&nbsp;  question: "What is caste system according to Ambedkar?"

```



\### SEMRAG Demonstration:

```bash

python demo.py

\# Shows complete SEMRAG pipeline with 5-step demonstration

```



\### Programmatic Usage (SEMRAG):

```python

from semrag.pipeline.ambedkargpt import AmbedkarGPT



pipeline = AmbedkarGPT()

pipeline.process\_document("data/Ambedkar\_book.pdf")

response = pipeline.answer\_question("What is caste system?", method="hybrid")

print(response\['answer'])

```



\## Project Evolution



\### Phase 1: Basic RAG

\- Simple Q\&A system for Ambedkar's speeches

\- Vector similarity retrieval

\- Basic evaluation metrics



\### Phase 2: Enhanced AmbedkarGPT

\- Cultural context preservation

\- AWS Bedrock integration

\- Knowledge graph enhancement

\- Comprehensive evaluation suite



\### Phase 3: SEMRAG Implementation

\- Research paper implementation

\- Semantic chunking (Algorithm 1)

\- Dual retrieval system (Equations 4 \& 5)

\- Community detection

\- Complete SEMRAG pipeline



\## Requirements



\### Core Dependencies:

```

\# Original system

langchain==0.1.0

sentence-transformers==2.2.2

chromadb==0.4.22

spacy==3.7.0



\# SEMRAG additions

networkx==3.1

python-louvain==0.16

ollama==0.1.4

pypdf==3.17.0

```



\### LLM Requirements:

\- \*\*Ollama\*\* with \*\*Mistral\*\* model (local)

\- Optional: AWS Bedrock credentials (for original system)



\## Testing



\### Run Original Tests:

```bash

python evaluation.py

```



\### Run Cultural Benchmark:

```bash

python run\_cultural\_benchmark.py 3

```



\### Test SEMRAG Components:

```bash

\# Test semantic chunking

python -c "from semrag.chunking.semantic\_chunker import test\_chunker; test\_chunker()"



\# Test knowledge graph

python -c "from semrag.graph.graph\_builder import test\_graph\_builder; test\_graph\_builder()"



\# Test retrieval

python -c "from semrag.retrieval.local\_search import test\_local\_search; test\_local\_search()"

```



\## Results



\### Original AmbedkarGPT:

\- \*\*Cultural Score\*\*: 0.333 (Basic Understanding)

\- \*\*Hit Rate\*\*: 88%

\- \*\*Best Strategy\*\*: Summarization (0.259 info loss)



\### SEMRAG Implementation:

\- \*\*Processing Time\*\*: 1.29s for 94 pages

\- \*\*Graph Size\*\*: 1720 nodes, 3175 edges

\- \*\*Community Detection\*\*: 9 thematic groups

\- \*\*Answer Quality\*\*: Research-paper compliant



\## Contributing



This project demonstrates:

1\. \*\*Production RAG system\*\* with cultural awareness

2\. \*\*Research implementation\*\* of SEMRAG architecture

3\. \*\*Evolution\*\* from basic to advanced RAG systems



\## References



1\. \*\*SEMRAG Research Paper\*\*: "Semantic Knowledge-Augmented RAG for Improved Question-Answering"

2\. \*\*Dr. B.R. Ambedkar\*\*: "Annihilation of Caste with a Reply to Mahatma Gandhi"

3\. \*\*AWS Bedrock Documentation\*\*

4\. \*\*LangChain RAG Framework\*\*





---



\*\*Developer\*\*: KATHERAPALLI UMA SHANKAR REDDY  

\*\*Date\*\*: 07/12/2025  

\*\*Repository\*\*: https://github.com/Uma-2005-sketch/AmbedkarGPT  

```





