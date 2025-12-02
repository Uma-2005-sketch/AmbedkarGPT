"""
Contextual Compression Analysis for Cultural Content
Measures information loss across different processing strategies.
"""

import json
import time
from typing import Dict, List, Tuple
import numpy as np
from collections import Counter

class CompressionAnalyzer:
    """Analyzes information loss in cultural text processing"""
    
    def __init__(self):
        self.metrics_history = []
        
        # Cultural content samples from Ambedkar speeches
        self.cultural_samples = [
            {
                "id": 1,
                "text": "The caste system is not merely a division of labor but a division of laborers based on birth. It is a hierarchy in which the divisions are graded one above the other in social status.",
                "cultural_elements": ["caste", "hierarchy", "birth", "social_status"],
                "complexity": "high"
            },
            {
                "id": 2,
                "text": "Untouchability is the worst form of social exclusion practiced in Indian society, where certain communities are denied access to water sources, temples, and public spaces.",
                "cultural_elements": ["untouchability", "social_exclusion", "discrimination", "access_denial"],
                "complexity": "high"
            },
            {
                "id": 3,
                "text": "Buddha's Dhamma is not merely a religion but a social gospel that emphasizes moral conduct, compassion, and equality among all beings.",
                "cultural_elements": ["buddhism", "dhamma", "social_gospel", "morality", "equality"],
                "complexity": "medium"
            },
            {
                "id": 4,
                "text": "The Constitution of India provides fundamental rights to all citizens regardless of caste, creed, or religion, aiming to establish a society based on justice and equality.",
                "cultural_elements": ["constitution", "fundamental_rights", "equality", "social_justice"],
                "complexity": "medium"
            }
        ]
    
    def calculate_semantic_similarity(self, original: str, processed: str) -> float:
        """Calculate semantic similarity between original and processed text"""
        # Simple word overlap approach (in production, use embeddings)
        original_words = set(original.lower().split())
        processed_words = set(processed.lower().split())
        
        if not original_words or not processed_words:
            return 0.0
        
        intersection = original_words.intersection(processed_words)
        union = original_words.union(processed_words)
        
        return len(intersection) / len(union)
    
    def calculate_information_loss(self, original: str, processed: str) -> Dict:
        """Calculate various information loss metrics"""
        
        # Length-based loss
        original_len = len(original.split())
        processed_len = len(processed.split())
        length_ratio = processed_len / original_len if original_len > 0 else 0
        
        # Keyword preservation
        cultural_keywords = ["caste", "untouchability", "buddha", "dhamma", 
                           "constitution", "rights", "equality", "hierarchy"]
        
        original_keywords = sum(1 for kw in cultural_keywords if kw in original.lower())
        processed_keywords = sum(1 for kw in cultural_keywords if kw in processed.lower())
        keyword_preservation = processed_keywords / original_keywords if original_keywords > 0 else 0
        
        # Semantic similarity
        semantic_similarity = self.calculate_semantic_similarity(original, processed)
        
        # Cultural element preservation
        cultural_elements = ["caste", "hierarchy", "exclusion", "discrimination", 
                           "buddhism", "morality", "rights", "justice"]
        original_elements = sum(1 for el in cultural_elements if el in original.lower())
        processed_elements = sum(1 for el in cultural_elements if el in processed.lower())
        cultural_preservation = processed_elements / original_elements if original_elements > 0 else 0
        
        return {
            "length_ratio": round(length_ratio, 3),
            "keyword_preservation": round(keyword_preservation, 3),
            "semantic_similarity": round(semantic_similarity, 3),
            "cultural_preservation": round(cultural_preservation, 3),
            "overall_loss": round(1 - ((keyword_preservation + semantic_similarity + cultural_preservation) / 3), 3)
        }
    
    def simulate_llm_processing(self, text: str, strategy: str = "summarize") -> str:
        """Simulate different LLM processing strategies"""
        
        # In production, this would call actual LLMs
        # For now, we simulate different compression strategies
        
        words = text.split()
        
        if strategy == "summarize":
            # Simulate summarization (keep first and last parts)
            if len(words) > 20:
                return " ".join(words[:10] + ["..."] + words[-10:])
            else:
                return text
        
        elif strategy == "extract_keypoints":
            # Simulate keypoint extraction
            key_words = [w for w in words if len(w) > 6 and w.lower() in 
                        ["caste", "system", "untouchability", "buddha", 
                         "constitution", "rights", "equality"]]
            if key_words:
                return f"Key points about {', '.join(key_words[:3])}: " + " ".join(words[:15])
            else:
                return " ".join(words[:15])
        
        elif strategy == "paraphrase":
            # Simulate paraphrasing (every other word)
            return " ".join(words[::2])
        
        else:
            return text
    
    def analyze_sample(self, sample: Dict, strategies: List[str] = None) -> Dict:
        """Analyze a cultural sample across different processing strategies"""
        if strategies is None:
            strategies = ["summarize", "extract_keypoints", "paraphrase"]
        
        results = []
        original_text = sample["text"]
        
        print(f"\nAnalyzing Sample {sample['id']} ({sample['complexity']} complexity):")
        print(f"Original: {original_text[:80]}...")
        
        for strategy in strategies:
            processed_text = self.simulate_llm_processing(original_text, strategy)
            
            loss_metrics = self.calculate_information_loss(original_text, processed_text)
            
            result = {
                "sample_id": sample["id"],
                "strategy": strategy,
                "original_length": len(original_text.split()),
                "processed_length": len(processed_text.split()),
                "processed_preview": processed_text[:80] + ("..." if len(processed_text) > 80 else ""),
                "loss_metrics": loss_metrics
            }
            
            results.append(result)
            
            print(f"  {strategy}: {loss_metrics['overall_loss']:.3f} loss "
                  f"({loss_metrics['cultural_preservation']:.3f} cultural preservation)")
        
        return results
    
    def run_analysis(self, num_samples: int = None) -> Dict:
        """Run comprehensive compression analysis"""
        print("="*60)
        print("CONTEXTUAL COMPRESSION ANALYSIS")
        print("Cultural Content Processing Information Loss")
        print("="*60)
        
        if num_samples is None or num_samples > len(self.cultural_samples):
            num_samples = len(self.cultural_samples)
        
        samples_to_analyze = self.cultural_samples[:num_samples]
        
        all_results = []
        strategy_performance = {}
        
        for sample in samples_to_analyze:
            sample_results = self.analyze_sample(sample)
            all_results.extend(sample_results)
            
            # Aggregate by strategy
            for result in sample_results:
                strategy = result["strategy"]
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {
                        "total_loss": 0,
                        "total_cultural": 0,
                        "count": 0
                    }
                
                strategy_performance[strategy]["total_loss"] += result["loss_metrics"]["overall_loss"]
                strategy_performance[strategy]["total_cultural"] += result["loss_metrics"]["cultural_preservation"]
                strategy_performance[strategy]["count"] += 1
        
        # Calculate averages
        print("\n" + "="*60)
        print("STRATEGY PERFORMANCE SUMMARY")
        print("="*60)
        
        for strategy, stats in strategy_performance.items():
            avg_loss = stats["total_loss"] / stats["count"]
            avg_cultural = stats["total_cultural"] / stats["count"]
            
            print(f"\n{strategy.upper()}:")
            print(f"  Average Information Loss: {avg_loss:.3f}")
            print(f"  Cultural Preservation: {avg_cultural:.3f}")
            print(f"  Samples analyzed: {stats['count']}")
            
            # Add to strategy performance
            strategy_performance[strategy]["avg_loss"] = avg_loss
            strategy_performance[strategy]["avg_cultural"] = avg_cultural
        
        # Overall statistics
        all_losses = [r["loss_metrics"]["overall_loss"] for r in all_results]
        all_cultural = [r["loss_metrics"]["cultural_preservation"] for r in all_results]
        
        overall_stats = {
            "total_samples": len(samples_to_analyze),
            "total_strategies": len(strategy_performance),
            "avg_information_loss": round(np.mean(all_losses), 3),
            "avg_cultural_preservation": round(np.mean(all_cultural), 3),
            "max_loss": round(max(all_losses), 3),
            "min_loss": round(min(all_losses), 3),
            "strategy_performance": strategy_performance
        }
        
        print("\n" + "="*60)
        print("OVERALL ANALYSIS")
        print("="*60)
        print(f"Total samples analyzed: {overall_stats['total_samples']}")
        print(f"Average information loss: {overall_stats['avg_information_loss']}")
        print(f"Average cultural preservation: {overall_stats['avg_cultural_preservation']}")
        print(f"Worst-case loss: {overall_stats['max_loss']}")
        print(f"Best-case loss: {overall_stats['min_loss']}")
        
        # Save results
        output = {
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_stats": overall_stats,
            "detailed_results": all_results,
            "cultural_samples": samples_to_analyze
        }
        
        filename = f"compression_analysis_{int(time.time())}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Full analysis saved to: {filename}")
        
        # Recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        best_strategy = min(strategy_performance.items(), 
                          key=lambda x: x[1]["avg_loss"])
        worst_strategy = max(strategy_performance.items(), 
                           key=lambda x: x[1]["avg_loss"])
        
        print(f"✅ Best strategy: {best_strategy[0]} "
              f"(loss: {best_strategy[1]['avg_loss']:.3f}, "
              f"cultural: {best_strategy[1]['avg_cultural']:.3f})")
        
        print(f"❌ Worst strategy: {worst_strategy[0]} "
              f"(loss: {worst_strategy[1]['avg_loss']:.3f}, "
              f"cultural: {worst_strategy[1]['avg_cultural']:.3f})")
        
        if overall_stats['avg_cultural_preservation'] < 0.5:
            print("\n⚠ WARNING: High cultural information loss detected!")
            print("   Consider implementing cultural context preservation techniques.")
        
        return output

def main():
    """Main function"""
    print("🔍 Contextual Compression Analysis")
    print("Measuring information loss in cultural content processing\n")
    
    try:
        analyzer = CompressionAnalyzer()
        
        # Ask how many samples to analyze
        import sys
        if len(sys.argv) > 1:
            try:
                num_samples = int(sys.argv[1])
            except:
                num_samples = 2  # Default
        else:
            num_samples = 2  # Quick test
        
        print(f"Analyzing {num_samples} cultural content samples...")
        print("This simulates different LLM processing strategies.\n")
        
        results = analyzer.run_analysis(num_samples=num_samples)
        
        print("\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()