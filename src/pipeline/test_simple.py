"""
Simple test for Jat-Pat-Todak Mandal question
"""

from ambedkargpt_enhanced import AmbedkarGPT

print("Testing Jat-Pat-Todak Mandal question...")
print("="*70)

ambedkargpt = AmbedkarGPT()

question = "Why did the Jat-Pat-Todak Mandal reject Ambedkar's speech?"
print(f"Question: {question}")
print("-"*70)

results = ambedkargpt.combined_search(question)

print(f"Found {results['search_metrics']['total_unique_chunks']} chunks")

if results['combined_results']:
    print("\nTop 3 chunks:")
    for i, chunk in enumerate(results['combined_results'][:3]):
        print(f"\n{i+1}. Score: {chunk['score']:.3f}")
        print(f"   {chunk['text'][:150]}...")
    
    print("\n" + "="*70)
    print("Generating answer...")
    answer = ambedkargpt.generate_answer(question, results['combined_results'])
    print("\nANSWER:")
    print(answer)
else:
    print("\nNo chunks found!")