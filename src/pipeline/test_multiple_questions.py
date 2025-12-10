"""
Testing AmbedkarGPT with multiple questions
"""

print("Testing AmbedkarGPT with multiple questions")
print("="*70)

from ambedkargpt_enhanced import AmbedkarGPT

ambedkargpt = AmbedkarGPT()

questions = [
    "What is caste according to Ambedkar?",
    "How does Ambedkar view Hindu society?",
    "What is the role of endogamy in caste formation?"
    "What are the characteristics of caste system?",
    "Who are the scholars mentioned by Ambedkar?",
    "How does caste differ from class?"
]

for question in questions:
    print(f"\nQuestion: {question}")
    print("-"*70)
    
    results = ambedkargpt.combined_search(question)
    print(f"Found {results['search_metrics']['total_unique_chunks']} chunks")
    
    if results['combined_results']:
        answer = ambedkargpt.generate_answer(question, results['combined_results'])
        print(f"\nAnswer: {answer[:300]}...")
    
    print()

print("="*70)
print("Testing completed!")