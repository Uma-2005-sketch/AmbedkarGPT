"""
LLM Client for SEMRAG system
Uses Ollama with fallback to mock responses
"""
import json
import time
from typing import Dict, List, Optional

class LLMClient:
    """
    Client for interacting with LLMs (Ollama, with mock fallback)
    """
    
    def __init__(self, model="mistral", temperature=0.1, max_tokens=1000):
        """
        Args:
            model: LLM model name (mistral, llama3, etc.)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ollama_available = self._check_ollama()
        
    def _check_ollama(self):
        """Check if Ollama is available"""
        try:
            import ollama
            # Try to list models to verify connection
            models = ollama.list()
            print(f"✅ Ollama available with models: {[m['name'] for m in models.get('models', [])]}")
            return True
        except ImportError:
            print("❌ Ollama not installed. Using mock responses.")
            return False
        except Exception as e:
            print(f"⚠️  Ollama error: {e}. Using mock responses.")
            return False
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate response from LLM
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Generated response
        """
        if self.ollama_available:
            return self._generate_with_ollama(prompt, system_prompt)
        else:
            return self._generate_mock_response(prompt, system_prompt)
    
    def _generate_with_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response using Ollama"""
        try:
            import ollama
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            )
            
            return response['message']['content']
            
        except Exception as e:
            print(f"❌ Ollama generation failed: {e}")
            return self._generate_mock_response(prompt, system_prompt)
    
    def _generate_mock_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate mock response when LLM is unavailable"""
        # Simple rule-based responses for Ambedkar topics
        prompt_lower = prompt.lower()
        
        response_templates = {
            'caste': "Dr. B.R. Ambedkar viewed the caste system as a hierarchical structure that divides society into watertight compartments. He argued for its complete annihilation through the rejection of religious scriptures that sanction caste.",
            'equality': "Ambedkar believed in complete social and political equality. He embedded this principle in the Indian Constitution through Articles 14-18, guaranteeing equality before law and prohibiting discrimination.",
            'democracy': "For Ambedkar, democracy was not just political but also social and economic. He emphasized liberty, equality, and fraternity as the pillars of a true democracy.",
            'constitution': "As Chairman of the Drafting Committee, Ambedkar played a key role in creating the Indian Constitution. He ensured it included protections for marginalized communities and fundamental rights for all citizens.",
            'ambedkar': "Dr. B.R. Ambedkar (1891-1956) was an Indian jurist, economist, and social reformer. He campaigned against social discrimination of Dalits and was the principal architect of the Indian Constitution.",
            'default': "Based on Ambedkar's works: He advocated for social justice, equality, and the annihilation of caste. His philosophy emphasized liberty, equality, and fraternity as fundamental principles for a just society."
        }
        
        # Check which template to use
        response = response_templates['default']
        for keyword, template in response_templates.items():
            if keyword in prompt_lower:
                response = template
                break
        
        # Add context from system prompt if provided
        if system_prompt:
            response = f"[Context: {system_prompt[:100]}...]\n\n{response}"
        
        return response
    
    def summarize(self, text: str, max_length: int = 200) -> str:
        """Summarize text using LLM"""
        prompt = f"Summarize the following text in {max_length} words or less:\n\n{text}"
        
        if self.ollama_available:
            return self._generate_with_ollama(prompt)
        else:
            # Simple truncation for mock
            words = text.split()
            if len(words) > max_length:
                return ' '.join(words[:max_length]) + '...'
            return text
    
    def extract_key_points(self, text: str, num_points: int = 3) -> List[str]:
        """Extract key points from text"""
        prompt = f"Extract {num_points} key points from the following text:\n\n{text}"
        
        if self.ollama_available:
            response = self._generate_with_ollama(prompt)
            # Parse bullet points or numbered list
            points = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('*') or 
                           (line[0].isdigit() and line[1] in ['.', ')'])):
                    points.append(line[2:].strip() if len(line) > 2 else line)
            
            if points:
                return points[:num_points]
        
        # Mock extraction
        sentences = text.split('. ')
        return [s.strip() + '.' for s in sentences[:num_points] if s.strip()]

# Test function
def test_llm_client():
    """Test LLM client"""
    print("LLM CLIENT TEST")
    print("=" * 50)
    
    # Create client
    client = LLMClient(model="mistral", temperature=0.1)
    
    # Test prompts
    test_prompts = [
        "Who was Dr. B.R. Ambedkar?",
        "What was Ambedkar's view on caste system?",
        "Explain the concept of equality in Ambedkar's philosophy."
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*50}")
        print(f"Test {i}: {prompt}")
        print('='*50)
        
        # Generate response
        start_time = time.time()
        response = client.generate(prompt)
        elapsed = time.time() - start_time
        
        print(f"\nResponse ({elapsed:.2f}s):")
        print("-" * 30)
        print(response[:300] + "..." if len(response) > 300 else response)
    
    # Test summarization
    print(f"\n{'='*50}")
    print("SUMMARIZATION TEST")
    print('='*50)
    
    sample_text = """
    Dr. B.R. Ambedkar was born on 14 April 1891 in Mhow, Central Provinces. 
    He was an Indian jurist, economist, and social reformer who campaigned 
    against social discrimination towards Dalits. He was the principal architect 
    of the Indian Constitution. Ambedkar published "Annihilation of Caste" in 1936, 
    which criticized the caste system and called for its destruction.
    """
    
    summary = client.summarize(sample_text, max_length=50)
    print(f"\nOriginal ({len(sample_text.split())} words):")
    print(sample_text[:200] + "...")
    
    print(f"\nSummary ({len(summary.split())} words):")
    print(summary)
    
    # Test key point extraction
    print(f"\n{'='*50}")
    print("KEY POINT EXTRACTION TEST")
    print('='*50)
    
    key_points = client.extract_key_points(sample_text, num_points=3)
    print("\nKey points:")
    for j, point in enumerate(key_points, 1):
        print(f"{j}. {point}")
    
    print("\n✅ LLM Client testing complete!")
    return client

if __name__ == "__main__":
    test_llm_client()