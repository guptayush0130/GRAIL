from openai import OpenAI
import networkx as nx

try:
    from config import OPENAI_API_KEY
except ImportError:
    print("ERROR: config.py not found or OPENAI_API_KEY is missing.")
    exit()

client = OpenAI(api_key=OPENAI_API_KEY)

def generate(user_query: str, ranked_nodes: dict, graph: nx.Graph, top_n: int = 5) -> str:
    
    print(f"[Synthesizer] Generating final answer using top {top_n} nodes...")
    
    if not ranked_nodes:
        return "I couldn't generate an answer due to insufficient data."
    
    # Get top N nodes
    top_nodes = list(ranked_nodes.keys())[:top_n]
    print(f"[Synthesizer] Using nodes: {top_nodes}")
    
    # Build context from top nodes
    context_parts = []
    for i, node in enumerate(top_nodes, 1):
        score = ranked_nodes[node]
        context = graph.nodes[node].get('combined_context', '')    
        context_parts.append(f"[Entity {i}: {node} (Importance: {score:.4f})]\n{context}")
    
    combined_context = "\n\n".join(context_parts)
    
    prompt = f"""You are a research analyst AI. Your task is to provide a comprehensive, well-structured answer to the user's question.

User's Question: {user_query}

You have been provided with information about the most important entities related to this question, ranked by relevance. You MUST base your answer ONLY on the information provided below.

Key Entities (Ranked by Importance):
{combined_context}

Instructions:
1. Write a comprehensive, multi-paragraph answer (3-5 paragraphs)
2. Synthesize information across all entities to tell a complete story
3. Start with the most important entity and its relevance to the query
4. Include specific details, facts, and relationships mentioned in the data
5. Address all parts of the user's question
6. Use natural, professional language
7. Do NOT make up information - only use what's provided
8. If certain aspects of the question can't be answered with the provided data, acknowledge this briefly

Generate your answer:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4 for high-quality synthesis
            messages=[
                {"role": "system", "content": "You are a professional research analyst who synthesizes information into clear, comprehensive answers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        final_answer = response.choices[0].message.content.strip()
        print(f"[Synthesizer] Generated answer ({len(final_answer)} characters)")
        return final_answer
        
    except Exception as e:
        print(f"[Synthesizer] ERROR: Failed to generate answer: {e}")
        return f"Error generating answer: {str(e)}"


# --- Main execution block for testing ---
if __name__ == "__main__":
    print("--- Running Synthesizer Test ---\n")
    
    # Create a test graph
    G = nx.Graph()
    
    nodes_data = {
        "Nvidia": "Nvidia leads the AI chip market with their H100 and H200 GPUs, which are used for training large language models. They announced the new Blackwell architecture at GTC 2025. Revenue from AI and datacenter grew 400% year-over-year.",
        "AMD": "AMD competes with Nvidia using their MI300X accelerators. They offer competitive pricing and have partnerships with Microsoft Azure and Meta. Market share is growing but still behind Nvidia.",
        "Microsoft": "Microsoft Azure uses Nvidia GPUs extensively for their AI services including Azure OpenAI. They are also developing their own Maia AI chips to reduce dependence on Nvidia.",
        "OpenAI": "OpenAI trained GPT-4 and ChatGPT using thousands of Nvidia A100 and H100 GPUs. They have a close partnership with Microsoft who provides compute infrastructure.",
        "Jensen Huang": "Nvidia CEO who has led the company's focus on AI. Known for keynote presentations where he announces new products and partnerships.",
    }
    
    for node, context in nodes_data.items():
        G.add_node(node, combined_context=context)
    
    # Mock ranked nodes (as if from PageRank)
    ranked_nodes = {
        "Nvidia": 0.35,
        "AMD": 0.20,
        "Microsoft": 0.18,
        "OpenAI": 0.15,
        "Jensen Huang": 0.12,
    }
    
    test_query = "Give me the latest products, performance, competitors, opportunities, and key collaborators of Nvidia. What things are going to affect it the most?"
    
    # Generate answer
    answer = generate(test_query, ranked_nodes, G, top_n=5)
    
    print("\n--- TEST RESULT ---")
    print(f"Query: {test_query}\n")
    print("Generated Answer:")
    print(answer)

