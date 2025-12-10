"""
Prompt templates for SEMRAG system
"""
from typing import Dict, List, Any

class PromptTemplates:
    """Collection of prompt templates for different tasks"""
    
    # System prompts
    SYSTEM_PROMPTS = {
        "ambedkar_expert": """You are an expert on Dr. B.R. Ambedkar's philosophy and works.
You are answering questions based on his book "Annihilation of Caste with a Reply to Mahatma Gandhi".
Always be accurate, cite sources when possible, and maintain cultural sensitivity.
If you don't know something based on the provided context, say so clearly.""",
        
        "general_qa": """You are a helpful AI assistant specialized in answering questions 
about historical texts and social philosophy. Provide accurate, well-reasoned responses."""
    }
    
    @staticmethod
    def local_rag_prompt(query: str, retrieved_items: List[Dict], context: str = "") -> str:
        """Prompt for local RAG search results"""
        context_str = ""
        if context:
            context_str = f"Previous context: {context}\n\n"
        
        entities_str = ""
        if retrieved_items:
            entities = set()
            for item in retrieved_items[:5]:  # Top 5 entities
                entities.add(f"{item.get('entity_label', '')} ({item.get('entity_type', '')})")
            
            if entities:
                entities_str = f"Relevant entities mentioned: {', '.join(entities)}\n\n"
        
        chunks_str = ""
        chunk_texts = []
        for item in retrieved_items[:3]:  # Top 3 chunks
            chunk_text = item.get('chunk_text', '')
            if chunk_text and chunk_text not in chunk_texts:
                chunk_texts.append(chunk_text)
        
        if chunk_texts:
            chunks_str = "Relevant text passages:\n"
            for i, text in enumerate(chunk_texts, 1):
                chunks_str += f"{i}. {text}\n"
            chunks_str += "\n"
        
        prompt = f"""{context_str}{entities_str}{chunks_str}
Based on the above information, answer the following question:

Question: {query}

Provide a comprehensive answer that synthesizes the information. 
If the information is insufficient, indicate what additional context would be helpful.

Answer:"""
        
        return prompt
    
    @staticmethod
    def global_rag_prompt(query: str, community_summaries: List[Dict], 
                         retrieved_points: List[Dict]) -> str:
        """Prompt for global RAG search results"""
        # Community summaries
        communities_str = ""
        if community_summaries:
            communities_str = "Thematic Communities Analysis:\n"
            for i, comm in enumerate(community_summaries[:3], 1):
                summary = comm.get('summary', '')
                if len(summary) > 200:
                    summary = summary[:200] + "..."
                communities_str += f"{i}. {summary}\n"
            communities_str += "\n"
        
        # Retrieved points
        points_str = ""
        if retrieved_points:
            points_str = "Key Information Points:\n"
            for i, point in enumerate(retrieved_points[:5], 1):
                text = point.get('chunk_text', '')
                if len(text) > 150:
                    text = text[:150] + "..."
                points_str += f"{i}. {text} (Relevance: {point.get('combined_score', 0):.2f})\n"
            points_str += "\n"
        
        prompt = f"""{communities_str}{points_str}
Based on the thematic analysis above, answer the following question:

Question: {query}

Provide an answer that:
1. Synthesizes information from different thematic communities
2. Shows understanding of the broader context
3. Cites relevant information from the key points

Answer:"""
        
        return prompt
    
    @staticmethod
    def hybrid_rag_prompt(query: str, local_results: Dict, global_results: Dict) -> str:
        """Prompt combining local and global RAG results"""
        # Extract local information
        local_entities = set()
        local_chunks = []
        
        if local_results and 'retrieved_items' in local_results:
            for item in local_results['retrieved_items'][:3]:
                local_entities.add(item.get('entity_label', ''))
                if 'chunk_text' in item:
                    local_chunks.append(item['chunk_text'][:100] + "...")
        
        # Extract global information
        global_communities = []
        global_points = []
        
        if global_results and 'retrieved_points' in global_results:
            for point in global_results['retrieved_points'][:3]:
                if 'community_summary' in point:
                    global_communities.append(point['community_summary'][:100] + "...")
                if 'chunk_text' in point:
                    global_points.append(point['chunk_text'][:100] + "...")
        
        # Build prompt
        prompt_parts = []
        
        if local_entities:
            prompt_parts.append(f"Key Entities: {', '.join(local_entities)}")
        
        if local_chunks:
            prompt_parts.append("Local Context:")
            for i, chunk in enumerate(local_chunks, 1):
                prompt_parts.append(f"{i}. {chunk}")
        
        if global_communities:
            prompt_parts.append("\nThematic Context:")
            for i, comm in enumerate(global_communities, 1):
                prompt_parts.append(f"{i}. {comm}")
        
        if global_points:
            prompt_parts.append("\nSupporting Information:")
            for i, point in enumerate(global_points, 1):
                prompt_parts.append(f"{i}. {point}")
        
        context = "\n".join(prompt_parts)
        
        prompt = f"""{context}

Based on both detailed entity analysis and broader thematic context, answer:

Question: {query}

Your answer should:
1. Address specific entities mentioned
2. Place them in broader thematic context  
3. Synthesize local and global perspectives
4. Provide a nuanced, comprehensive response

Answer:"""
        
        return prompt
    
    @staticmethod
    def community_summary_prompt(nodes: List[str], relations: List[Dict]) -> str:
        """Prompt for generating community summaries"""
        nodes_str = "\n".join([f"- {node}" for node in nodes[:10]])
        relations_str = "\n".join([f"- {r.get('subject', '')} {r.get('relation', '')} {r.get('object', '')}" 
                                 for r in relations[:5]])
        
        prompt = f"""Analyze the following knowledge graph community:

Nodes in this community:
{nodes_str}

Key relations:
{relations_str}

Generate a concise summary (2-3 sentences) that captures:
1. The main theme or topic of this community
2. Key entities and their relationships
3. The significance or implications

Summary:"""
        
        return prompt

# Test function
def test_prompt_templates():
    """Test prompt templates"""
    print("PROMPT TEMPLATES TEST")
    print("=" * 50)
    
    templates = PromptTemplates()
    
    # Test data
    local_items = [
        {
            'entity_label': 'Ambedkar',
            'entity_type': 'person',
            'chunk_text': 'Ambedkar criticized the caste system as hierarchical.',
            'combined_score': 0.85
        },
        {
            'entity_label': 'caste',
            'entity_type': 'concept', 
            'chunk_text': 'The caste system divides society into compartments.',
            'combined_score': 0.78
        }
    ]
    
    global_points = [
        {
            'community_summary': 'Community about social reform and caste system',
            'chunk_text': 'Ambedkar advocated for annihilation of caste through religious reform.',
            'combined_score': 0.82
        }
    ]
    
    # Test local prompt
    print("\n1. LOCAL RAG PROMPT:")
    print("-" * 30)
    local_prompt = templates.local_rag_prompt(
        "What is caste system?", 
        local_items,
        "Previous discussion about Ambedkar"
    )
    print(local_prompt[:300] + "..." if len(local_prompt) > 300 else local_prompt)
    
    # Test global prompt  
    print("\n\n2. GLOBAL RAG PROMPT:")
    print("-" * 30)
    global_prompt = templates.global_rag_prompt(
        "What is caste system?",
        [{'summary': 'Social reform community'}],
        global_points
    )
    print(global_prompt[:300] + "..." if len(global_prompt) > 300 else global_prompt)
    
    # Test hybrid prompt
    print("\n\n3. HYBRID RAG PROMPT:")
    print("-" * 30)
    hybrid_prompt = templates.hybrid_rag_prompt(
        "What is caste system?",
        {'retrieved_items': local_items},
        {'retrieved_points': global_points}
    )
    print(hybrid_prompt[:300] + "..." if len(hybrid_prompt) > 300 else hybrid_prompt)
    
    print("\n✅ Prompt templates tested successfully!")
    return templates

if __name__ == "__main__":
    test_prompt_templates()