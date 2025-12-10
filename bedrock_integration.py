"""
AWS Bedrock integration for AmbedkarGPT.
"""

import json
import os
import sys

def get_bedrock_client():
    """
    Returns a Bedrock runtime client.
    Falls back to mock client if AWS credentials are not configured.
    """
    try:
        # Try to import real boto3
        import boto3
        from botocore.exceptions import NoCredentialsError, ClientError
        
        # Test credentials
        sts = boto3.client('sts', region_name='us-east-1')
        identity = sts.get_caller_identity()
        print(f"✅ AWS Identity: {identity['Arn']}")
        
        # Create Bedrock client
        client = boto3.client(
            'bedrock-runtime',
            region_name='us-east-1'
        )
        print("✅ Using real AWS Bedrock client")
        return client, "aws"
        
    except (NoCredentialsError, ClientError, Exception) as e:
        print(f"⚠ AWS credentials issue: {e}")
        print("🔄 Falling back to mock Bedrock (Ollama/Mistral)")
        
        # Use mock client
        from mock_bedrock import MockBedrockRuntime
        client = MockBedrockRuntime(region_name="us-east-1")
        return client, "mock"

class BedrockLLM:
    """Unified LLM interface for AWS Bedrock or local fallback"""
    
    def __init__(self, model_id="anthropic.claude-3-haiku-20240307-v1:0"):
        self.model_id = model_id
        self.client, self.client_type = get_bedrock_client()
        print(f"📦 Model: {model_id}")
    
    def invoke(self, prompt, max_tokens=500):
        """Invoke LLM with prompt"""
        
        if self.client_type == "aws":
            # Real AWS Bedrock invocation for Claude
            body = json.dumps({
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "max_tokens_to_sample": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
            })
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body
            )
            
            response_body = json.loads(response['body'].read())
            return response_body.get('completion', '').strip()
        
        else:
            # Mock client
            body = json.dumps({"prompt": prompt})
            response = self.client.invoke_model(body=body)
            response_body = json.loads(response['body'])
            return response_body.get('completion', '').strip()

def test_bedrock_connection():
    """Test Bedrock connection"""
    print("\n" + "="*50)
    print("Testing Bedrock Integration")
    print("="*50)
    
    llm = BedrockLLM()
    test_prompt = "What is the real remedy for caste system according to Ambedkar?"
    print(f"\n📝 Prompt: {test_prompt}")
    
    response = llm.invoke(test_prompt)
    print(f"\n✅ Response: {response[:200]}...")
    
    return True

if __name__ == "__main__":
    test_bedrock_connection()