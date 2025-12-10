"""
Run Cultural Context Benchmark on Enhanced AmbedkarGPT
"""

import json
import time
from typing import List, Dict
from cultural_evaluator import CulturalEvaluator

# Import our system (simplified version to avoid heavy initialization)
class BenchmarkRunner:
    """Lightweight runner for cultural benchmark"""
    
    def __init__(self):
        # We'll reuse existing system components
        try:
            from bedrock_integration import BedrockLLM
            self.llm = BedrockLLM()
            print("✅ LLM initialized for benchmark")
        except Exception as e:
            print(f"⚠ LLM initialization failed: {e}")
            self.llm = None
    
    def generate_answer(self, question: str) -> str:
        """Generate answer for benchmark question"""
        if not self.llm:
            return "LLM not available for benchmark"
        
        # Simplified prompt for benchmark
        prompt = f"""You are answering a cultural understanding benchmark question about Dr. B.R. Ambedkar and Indian social context.

Question: {question}

Provide a comprehensive, culturally sensitive answer that demonstrates understanding of historical and social context.
Focus on accuracy, nuance, and cultural awareness.

Answer: """
        
        try:
            answer = self.llm.invoke(prompt)
            return answer
        except Exception as e:
            return f"Error: {e}"
    
    def load_benchmark_questions(self, benchmark_file: str = "cultural_benchmark.json") -> List[Dict]:
        """Load benchmark questions"""
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            benchmark = json.load(f)
        
        questions = []
        for q in benchmark['questions']:
            questions.append({
                "question_id": q['id'],
                "question": q['question'],
                "type": q['type'],
                "expected_keywords": q.get('expected_keywords', []),
                "difficulty": q.get('difficulty', 'medium')
            })
        
        return questions
    
    def run(self, max_questions: int = 5):
        """Run cultural benchmark"""
        print("="*60)
        print("RUNNING CULTURAL CONTEXT BENCHMARK")
        print("="*60)
        
        # Load questions
        questions = self.load_benchmark_questions()
        
        if max_questions and max_questions < len(questions):
            questions = questions[:max_questions]
        
        print(f"Evaluating {len(questions)} questions...\n")
        
        # Generate answers
        qa_pairs = []
        for i, q in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] Processing: {q['question'][:50]}...")
            
            answer = self.generate_answer(q['question'])
            qa_pairs.append({
                "question_id": q['question_id'],
                "question": q['question'],
                "answer": answer,
                "type": q['type'],
                "difficulty": q['difficulty']
            })
            
            # Brief pause to avoid rate limiting
            time.sleep(2)
            
            # Show answer preview
            print(f"   Answer preview: {answer[:80]}...")
        
        # Evaluate results
        print("\n" + "="*60)
        print("EVALUATING CULTURAL UNDERSTANDING")
        print("="*60)
        
        evaluator = CulturalEvaluator()
        results = evaluator.run_benchmark(qa_pairs)
        
        # Save detailed results
        output_data = {
            "benchmark_run": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_questions": len(qa_pairs),
                "system": "Enhanced AmbedkarGPT (Bedrock + SEMRAG)"
            },
            "qa_pairs": qa_pairs,
            "evaluation_results": results
        }
        
        # Save to file
        output_file = f"cultural_benchmark_results_{int(time.time())}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Full results saved to: {output_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(f"System: Enhanced AmbedkarGPT")
        print(f"Model: Bedrock (Claude 3 mock with Ollama/Mistral fallback)")
        print(f"Questions: {len(qa_pairs)}")
        print(f"Average Cultural Score: {results['summary']['average_score']:.3f}")
        
        # Recommendation
        score = results['summary']['average_score']
        if score >= 0.7:
            print("✅ Excellent cultural understanding")
        elif score >= 0.5:
            print("⚠ Moderate cultural understanding")
        elif score >= 0.3:
            print("📉 Basic cultural understanding")
        else:
            print("❌ Limited cultural understanding")
        
        return results

def main():
    """Main function"""
    print("🚀 Cultural Context Benchmark for AmbedkarGPT")
    print("This will test the system's cultural understanding capabilities.")
    print("It may take a few minutes to complete.\n")
    
    try:
        runner = BenchmarkRunner()
        
        # Ask how many questions to run
        import sys
        if len(sys.argv) > 1:
            try:
                max_q = int(sys.argv[1])
            except:
                max_q = 3
        else:
            max_q = 3  # Default to 3 questions for quick test
        
        print(f"Running {max_q} questions from cultural benchmark...")
        
        results = runner.run(max_questions=max_q)
        
        print("\n✅ Benchmark complete!")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()