"""
Cultural Context Evaluator for AmbedkarGPT
Implements specialized metrics for cultural understanding evaluation.
"""

import json
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import Counter

@dataclass
class CulturalMetrics:
    """Container for cultural evaluation metrics"""
    cultural_relevance: float  # 0-1
    contextual_accuracy: float  # 0-1
    nuance_understanding: float  # 0-1
    sensitivity_score: float  # 0-1
    completeness_score: float  # 0-1
    
    @property
    def overall_score(self) -> float:
        """Calculate overall cultural understanding score"""
        weights = {
            'cultural_relevance': 0.25,
            'contextual_accuracy': 0.25,
            'nuance_understanding': 0.20,
            'sensitivity_score': 0.15,
            'completeness_score': 0.15
        }
        
        scores = [
            self.cultural_relevance * weights['cultural_relevance'],
            self.contextual_accuracy * weights['contextual_accuracy'],
            self.nuance_understanding * weights['nuance_understanding'],
            self.sensitivity_score * weights['sensitivity_score'],
            self.completeness_score * weights['completeness_score']
        ]
        
        return sum(scores)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "cultural_relevance": round(self.cultural_relevance, 3),
            "contextual_accuracy": round(self.contextual_accuracy, 3),
            "nuance_understanding": round(self.nuance_understanding, 3),
            "sensitivity_score": round(self.sensitivity_score, 3),
            "completeness_score": round(self.completeness_score, 3),
            "overall_score": round(self.overall_score, 3)
        }

class CulturalEvaluator:
    """Evaluates cultural understanding in LLM responses"""
    
    def __init__(self, benchmark_file="cultural_benchmark.json"):
        self.benchmark = self._load_benchmark(benchmark_file)
        
        # Cultural dictionaries
        self.cultural_terms = {
            "caste": ["varna", "jati", "brahmin", "kshatriya", "vaishya", "shudra", "dalit"],
            "buddhism": ["dhamma", "sangha", "nirvana", "middle_way", "four_noble_truths"],
            "democracy": ["constitution", "rights", "representation", "equality", "liberty"],
            "social_justice": ["equality", "rights", "empowerment", "discrimination", "exclusion"]
        }
        
        self.sensitive_terms = {
            "high": ["untouchable", "polluted", "impure", "inferior"],
            "medium": ["caste", "hierarchy", "discrimination"],
            "low": ["social", "cultural", "traditional"]
        }
    
    def _load_benchmark(self, benchmark_file: str) -> Dict:
        """Load cultural benchmark"""
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def evaluate_response(self, question_id: int, question: str, 
                         answer: str, expected_keywords: List[str]) -> CulturalMetrics:
        """Evaluate a single Q&A pair"""
        
        # 1. Cultural Relevance
        cultural_relevance = self._calculate_cultural_relevance(answer, expected_keywords)
        
        # 2. Contextual Accuracy
        contextual_accuracy = self._calculate_contextual_accuracy(answer)
        
        # 3. Nuance Understanding
        nuance_understanding = self._calculate_nuance_understanding(answer)
        
        # 4. Sensitivity Score
        sensitivity_score = self._calculate_sensitivity_score(answer)
        
        # 5. Completeness Score
        completeness_score = self._calculate_completeness_score(answer, expected_keywords)
        
        return CulturalMetrics(
            cultural_relevance=cultural_relevance,
            contextual_accuracy=contextual_accuracy,
            nuance_understanding=nuance_understanding,
            sensitivity_score=sensitivity_score,
            completeness_score=completeness_score
        )
    
    def _calculate_cultural_relevance(self, answer: str, expected_keywords: List[str]) -> float:
        """Calculate relevance to cultural context"""
        answer_lower = answer.lower()
        relevant_keywords = [kw for kw in expected_keywords if kw.lower() in answer_lower]
        
        if not expected_keywords:
            return 0.5  # Default if no keywords
        
        return len(relevant_keywords) / len(expected_keywords)
    
    def _calculate_contextual_accuracy(self, answer: str) -> float:
        """Calculate contextual accuracy based on cultural terminology"""
        score = 0.0
        total_terms = 0
        
        for category, terms in self.cultural_terms.items():
            category_score = 0
            for term in terms:
                if re.search(rf'\b{term}\b', answer.lower()):
                    category_score += 1
            
            if terms:  # Avoid division by zero
                score += category_score / len(terms)
                total_terms += 1
        
        return score / total_terms if total_terms > 0 else 0.0
    
    def _calculate_nuance_understanding(self, answer: str) -> float:
        """Calculate understanding of nuances"""
        # Check for nuance indicators
        nuance_indicators = [
            "however", "although", "while", "despite", "on the other hand",
            "context", "perspective", "interpretation", "complex", "multifaceted"
        ]
        
        answer_lower = answer.lower()
        found_indicators = sum(1 for indicator in nuance_indicators 
                              if indicator in answer_lower)
        
        # Normalize to 0-1 (max 5 indicators for full score)
        return min(found_indicators / 5, 1.0)
    
    def _calculate_sensitivity_score(self, answer: str) -> float:
        """Calculate cultural sensitivity score"""
        answer_lower = answer.lower()
        
        # Check for insensitive language
        insensitive_phrases = [
            "all hindus", "always", "never", "everyone believes",
            "simple", "easy solution", "just", "merely"
        ]
        
        insensitive_count = sum(1 for phrase in insensitive_phrases 
                               if phrase in answer_lower)
        
        # Check for sensitive handling
        sensitive_handlers = [
            "considered", "viewed as", "historically", "in context",
            "some believe", "many argue", "varied perspectives"
        ]
        
        sensitive_count = sum(1 for handler in sensitive_handlers 
                            if handler in answer_lower)
        
        # Calculate score
        if insensitive_count == 0 and sensitive_count > 0:
            return 1.0
        elif insensitive_count == 0:
            return 0.7
        elif insensitive_count <= 2:
            return 0.5
        else:
            return 0.2
    
    def _calculate_completeness_score(self, answer: str, expected_keywords: List[str]) -> float:
        """Calculate completeness of answer"""
        # Length-based completeness (minimum 50 words for good answer)
        word_count = len(answer.split())
        length_score = min(word_count / 100, 1.0)  # 100 words = full score
        
        # Keyword coverage (already calculated in cultural_relevance)
        keyword_score = self._calculate_cultural_relevance(answer, expected_keywords)
        
        # Combine scores
        return (length_score * 0.3) + (keyword_score * 0.7)
    
    def run_benchmark(self, qa_pairs: List[Dict]) -> Dict:
        """Run full benchmark evaluation"""
        results = []
        overall_scores = []
        
        print("="*60)
        print("CULTURAL CONTEXT BENCHMARK EVALUATION")
        print("="*60)
        
        for pair in qa_pairs:
            question_id = pair.get('question_id', 0)
            question = pair.get('question', '')
            answer = pair.get('answer', '')
            
            # Find benchmark question
            benchmark_q = next((q for q in self.benchmark['questions'] 
                              if q['id'] == question_id), None)
            
            if not benchmark_q:
                print(f"⚠ Question ID {question_id} not found in benchmark")
                continue
            
            expected_keywords = benchmark_q.get('expected_keywords', [])
            
            # Evaluate
            metrics = self.evaluate_response(
                question_id, question, answer, expected_keywords
            )
            
            results.append({
                "question_id": question_id,
                "question": question[:50] + "...",
                "metrics": metrics.to_dict(),
                "expected_keywords": expected_keywords,
                "matched_keywords": [kw for kw in expected_keywords 
                                   if kw.lower() in answer.lower()]
            })
            
            overall_scores.append(metrics.overall_score)
            
            # Print individual result
            print(f"\nQ{question_id}: {question[:40]}...")
            print(f"  Score: {metrics.overall_score:.3f}")
            print(f"  Relevance: {metrics.cultural_relevance:.3f}")
            print(f"  Accuracy: {metrics.contextual_accuracy:.3f}")
        
        # Calculate overall statistics
        if overall_scores:
            avg_score = sum(overall_scores) / len(overall_scores)
            max_score = max(overall_scores)
            min_score = min(overall_scores)
            
            print("\n" + "="*60)
            print("OVERALL RESULTS")
            print("="*60)
            print(f"Average Cultural Score: {avg_score:.3f}")
            print(f"Best Score: {max_score:.3f}")
            print(f"Worst Score: {min_score:.3f}")
            print(f"Questions Evaluated: {len(results)}")
        
        return {
            "results": results,
            "summary": {
                "average_score": avg_score if overall_scores else 0,
                "max_score": max_score if overall_scores else 0,
                "min_score": min_score if overall_scores else 0,
                "total_questions": len(results)
            }
        }
    
    def save_results(self, results: Dict, filename: str = "cultural_evaluation_results.json"):
        """Save evaluation results to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to {filename}")

def test_evaluator():
    """Test the cultural evaluator"""
    print("Testing Cultural Evaluator...")
    
    evaluator = CulturalEvaluator()
    
    # Test data
    test_qa_pairs = [
        {
            "question_id": 1,
            "question": "What is the cultural significance of the caste system?",
            "answer": "The caste system has deep cultural significance in Indian history, rooted in Hindu scriptures. It established social hierarchy through varna and jati systems, influencing social structure and religious practices for centuries."
        },
        {
            "question_id": 2,
            "question": "How did Ambedkar view Buddhism?",
            "answer": "Ambedkar saw Buddhism as a social gospel that rejected caste hierarchy. He viewed it as an alternative moral framework emphasizing equality and social justice, different from traditional Hindu practices."
        }
    ]
    
    results = evaluator.run_benchmark(test_qa_pairs)
    return results

if __name__ == "__main__":
    test_evaluator()