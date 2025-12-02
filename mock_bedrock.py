"""
Mock AWS Bedrock client for local development.
"""

import json
from langchain_community.llms import Ollama

class MockBedrockRuntime:
    def __init__(self, region_name="us-east-1"):
        self.region_name = region_name
        print(f"🔧 Mock Bedrock Runtime (region: {region_name})")
        print("📝 Using Ollama/Mistral as fallback")
        
        # Use Ollama as fallback
        self.llm = Ollama(model="mistral")
    
    def invoke_model(self, **kwargs):
        """Mock invoke_model that uses local Ollama"""
        try:
            body = kwargs.get('body', '{}')
            body_dict = json.loads(body)
            
            # Extract prompt from different possible formats
            prompt = ""
            if 'prompt' in body_dict:
                prompt = body_dict['prompt']
            elif 'inputText' in body_dict:
                prompt = body_dict['inputText']
            elif 'messages' in body_dict:
                # Claude message format
                messages = body_dict.get('messages', [])
                if messages:
                    prompt = messages[-1].get('content', '')
            
            if not prompt:
                prompt = "Hello, respond with a test message."
            
            # Call local Ollama
            response = self.llm.invoke(prompt)
            
            # Return in AWS-like format
            return {
                'body': json.dumps({
                    'completion': response,
                    'stop_reason': 'end_turn'
                })
            }
        except Exception as e:
            return {
                'body': json.dumps({
                    'completion': f"Error: {e}",
                    'stop_reason': 'error'
                })
            }

# Simple test
if __name__ == "__main__":
    client = MockBedrockRuntime()
    test_body = json.dumps({"prompt": "Hello, who are you?"})
    result = client.invoke_model(body=test_body)
    print("Test result:", result)