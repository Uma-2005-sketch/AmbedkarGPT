import json
import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.llms import Ollama
from rouge_score import rouge_scorer
import os

# Disable SSL verification
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'

class AmbedkarEvaluator:
    def __init__(self):
        self.llm = Ollama(model="mistral")
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.results = []
        
        # Try to import nltk for BLEU score, but handle if not available
        try:
            import nltk
            # Try to download punkt data
            try:
                nltk.download('punkt', quiet=True)
            except:
                # If download fails, create a simple tokenizer fallback
                pass
            self.nltk_available = True
        except:
            self.nltk_available = False
            print("⚠️  NLTK not available, BLEU score will be skipped")
        
    def load_documents(self, chunk_size=500):
        """Load all documents from corpus folder"""
        all_texts = []
        doc_mapping = {}
        
        for i in range(1, 7):
            filename = f"corpus/speech{i}.txt"
            try:
                loader = TextLoader(filename, encoding="utf-8")
                documents = loader.load()
                
                # Split text
                text_splitter = CharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=50,
                    separator="\n"
                )
                texts = text_splitter.split_documents(documents)
                
                for text in texts:
                    all_texts.append(text.page_content)
                    doc_mapping[len(all_texts)-1] = f"speech{i}.txt"
                    
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                
        return all_texts, doc_mapping
    
    def simple_retrieve(self, question, texts, top_k=3):
        """Simple text-based retrieval"""
        question_lower = question.lower()
        scores = []
        
        for i, text in enumerate(texts):
            text_lower = text.lower()
            # Simple word overlap scoring
            score = sum(1 for word in question_lower.split() if word in text_lower)
            scores.append((score, i, text))
        
        # Sort by score and return top_k
        scores.sort(reverse=True, key=lambda x: x[0])
        return [(text, idx) for score, idx, text in scores[:top_k] if score > 0]
    
    def get_answer(self, question, context):
        """Get answer from LLM given context"""
        prompt = f"""Based on the following context from Dr. B.R. Ambedkar's speeches, answer the question concisely.

Context: {context}

Question: {question}

Answer: """
        
        try:
            answer = self.llm.invoke(prompt)
            return answer.strip()
        except Exception as e:
            return f"Error: {e}"
    
    def calculate_rouge_l(self, generated, reference):
        """Calculate ROUGE-L score"""
        scores = self.rouge_scorer.score(reference, generated)
        return scores['rougeL'].fmeasure
    
    def calculate_bleu(self, generated, reference):
        """Calculate BLEU score with fallback"""
        if not self.nltk_available:
            return 0.0  # Return 0 if NLTK not available
            
        try:
            from nltk.translate.bleu_score import sentence_bleu
            import nltk
            
            # Simple tokenization fallback if punkt not available
            try:
                reference_tokens = [nltk.word_tokenize(reference.lower())]
                generated_tokens = nltk.word_tokenize(generated.lower())
            except:
                # Fallback: split by spaces
                reference_tokens = [reference.lower().split()]
                generated_tokens = generated.lower().split()
                
            return sentence_bleu(reference_tokens, generated_tokens)
        except Exception as e:
            print(f"⚠️  BLEU calculation failed: {e}")
            return 0.0
    
    def calculate_cosine_similarity(self, generated, reference):
        """Calculate cosine similarity between sentences"""
        # Simple word vector approach
        gen_words = set(generated.lower().split())
        ref_words = set(reference.lower().split())
        
        intersection = gen_words.intersection(ref_words)
        union = gen_words.union(ref_words)
        
        if len(union) == 0:
            return 0.0
        return len(intersection) / len(union)
    
    def calculate_answer_relevance(self, generated, question):
        """Simple answer relevance check"""
        question_words = set(question.lower().split())
        answer_words = set(generated.lower().split())
        common_words = question_words.intersection(answer_words)
        
        if len(question_words) == 0:
            return 0.0
        return len(common_words) / len(question_words)
    
    def calculate_faithfulness(self, generated_answer, retrieved_context):
        """Calculate if the generated answer is faithful to the retrieved context"""
        if not generated_answer or not retrieved_context:
            return 0.0
        
        generated_lower = generated_answer.lower()
        context_lower = retrieved_context.lower()
        
        # Simple approach: check if key nouns from answer are in context
        # This is a simplified faithfulness check
        important_words = []
        for word in generated_lower.split():
            if len(word) > 4 and word not in ['based', 'according', 'document', 'speech']:
                important_words.append(word)
        
        if not important_words:
            return 0.0
        
        faithful_words = sum(1 for word in important_words if word in context_lower)
        return faithful_words / len(important_words)
    
    def evaluate_retrieval(self, retrieved_docs, expected_docs, doc_mapping):
        """Calculate retrieval metrics"""
        retrieved_filenames = [doc_mapping.get(idx, "unknown") for _, idx in retrieved_docs]
        
        # Hit Rate
        hit_rate = 1.0 if any(doc in retrieved_filenames for doc in expected_docs) else 0.0
        
        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        for i, doc in enumerate(retrieved_filenames):
            if doc in expected_docs:
                mrr = 1.0 / (i + 1)
                break
        
        # Precision@K
        k = min(3, len(retrieved_filenames))
        relevant_retrieved = sum(1 for doc in retrieved_filenames[:k] if doc in expected_docs)
        precision_at_k = relevant_retrieved / k if k > 0 else 0.0
        
        return {
            "hit_rate": hit_rate,
            "mrr": mrr,
            "precision_at_k": precision_at_k,
            "retrieved_docs": retrieved_filenames,
            "expected_docs": expected_docs
        }
    
    def run_evaluation(self, chunk_size=500):
        """Run comprehensive evaluation"""
        print(f"🚀 Starting evaluation with chunk size: {chunk_size}")
        
        # Load test dataset
        with open('test_dataset.json', 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # Load documents
        texts, doc_mapping = self.load_documents(chunk_size)
        print(f"📚 Loaded {len(texts)} text chunks from documents")
        
        results = []
        
        for item in test_data['test_questions']:
            print(f"🔍 Evaluating Q{item['id']}: {item['question'][:50]}...")
            
            # Retrieve relevant documents
            retrieved = self.simple_retrieve(item['question'], texts, top_k=3)
            context = "\n\n".join([text for text, _ in retrieved])
            
            # Get answer from LLM
            generated_answer = self.get_answer(item['question'], context)
            
            # Calculate metrics
            retrieval_metrics = self.evaluate_retrieval(
                retrieved, item['source_documents'], doc_mapping
            )
            
            # Only calculate answer metrics for answerable questions
            if item['answerable'] and item['ground_truth'] != "This information is not available in the provided documents.":
                rouge_l = self.calculate_rouge_l(generated_answer, item['ground_truth'])
                bleu = self.calculate_bleu(generated_answer, item['ground_truth'])
                cosine_sim = self.calculate_cosine_similarity(generated_answer, item['ground_truth'])
                answer_relevance = self.calculate_answer_relevance(generated_answer, item['question'])
                faithfulness = self.calculate_faithfulness(generated_answer, context)
            else:
                rouge_l = bleu = cosine_sim = answer_relevance = faithfulness = 0.0
            
            result = {
                "id": item['id'],
                "question": item['question'],
                "generated_answer": generated_answer,
                "ground_truth": item['ground_truth'],
                "retrieval_metrics": retrieval_metrics,
                "answer_metrics": {
                    "rouge_l": rouge_l,
                    "bleu": bleu,
                    "cosine_similarity": cosine_sim,
                    "answer_relevance": answer_relevance,
                    "faithfulness": faithfulness
                },
                "question_type": item['question_type'],
                "answerable": item['answerable']
            }
            
            results.append(result)
        
        return results
    
    def save_results(self, results, filename):
        """Save evaluation results to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to {filename}")

def main():
    evaluator = AmbedkarEvaluator()
    
    # Test different chunking strategies
    chunk_sizes = [300, 500, 800]  # Small, Medium, Large
    
    for chunk_size in chunk_sizes:
        print(f"\n{'='*60}")
        print(f"📊 Evaluating with chunk size: {chunk_size}")
        print(f"{'='*60}")
        
        results = evaluator.run_evaluation(chunk_size)
        
        # Calculate average metrics
        avg_retrieval = {
            "hit_rate": np.mean([r['retrieval_metrics']['hit_rate'] for r in results]),
            "mrr": np.mean([r['retrieval_metrics']['mrr'] for r in results]),
            "precision_at_k": np.mean([r['retrieval_metrics']['precision_at_k'] for r in results])
        }
        
        answerable_results = [r for r in results if r['answerable'] and r['ground_truth'] != "This information is not available in the provided documents."]
        if answerable_results:
            avg_answer = {
                "rouge_l": np.mean([r['answer_metrics']['rouge_l'] for r in answerable_results]),
                "bleu": np.mean([r['answer_metrics']['bleu'] for r in answerable_results]),
                "cosine_similarity": np.mean([r['answer_metrics']['cosine_similarity'] for r in answerable_results]),
                "answer_relevance": np.mean([r['answer_metrics']['answer_relevance'] for r in answerable_results]),
                "faithfulness": np.mean([r['answer_metrics']['faithfulness'] for r in answerable_results])
            }
        else:
            avg_answer = {"rouge_l": 0, "bleu": 0, "cosine_similarity": 0, "answer_relevance": 0, "faithfulness": 0}
        
        print(f"\n📈 Average Metrics for chunk size {chunk_size}:")
        print(f"Retrieval - Hit Rate: {avg_retrieval['hit_rate']:.3f}, MRR: {avg_retrieval['mrr']:.3f}, P@K: {avg_retrieval['precision_at_k']:.3f}")
        print(f"Answer - ROUGE-L: {avg_answer['rouge_l']:.3f}, BLEU: {avg_answer['bleu']:.3f}, Cosine: {avg_answer['cosine_similarity']:.3f}, Relevance: {avg_answer['answer_relevance']:.3f}, Faithfulness: {avg_answer['faithfulness']:.3f}")
        
        # Save results
        filename = f"test_results_chunk_{chunk_size}.json"
        evaluator.save_results(results, filename)

if __name__ == "__main__":
    main()