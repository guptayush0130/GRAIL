import networkx as nx
import featurizer

def run(graph: nx.Graph) -> dict:
    """
    The GRAIL ranking algorithm:
    1. Featurize all nodes (embed their combined_context)
    2. Weight all edges by cosine similarity
    3. Run PageRank on the weighted graph
    4. Return sorted nodes by importance
    
    Args:
        graph: A NetworkX graph where nodes have 'combined_context' attribute
        
    Returns:
        A dictionary mapping node names to their PageRank scores, sorted by score (highest first)
    """
    print("[Ranker] Starting GRAIL ranking algorithm...")
    
    if len(graph.nodes) == 0:
        print("[Ranker] WARNING: Empty graph provided")
        return {}
    
    print(f"[Ranker] Step 1: Featurizing {len(graph.nodes)} nodes...")
    for node in graph.nodes:
        context = graph.nodes[node].get('combined_context', '')
        if not context:
            print(f"[Ranker] WARNING: Node '{node}' has no combined_context")
            context = node 
        
        embedding = featurizer.embed(context)
        graph.nodes[node]['embedding'] = embedding
    
    print("[Ranker] All nodes featurized.")
    
    print(f"[Ranker] Step 2: Weighting {len(graph.edges)} edges...")
    for u, v in graph.edges:
        vec_u = graph.nodes[u]['embedding']
        vec_v = graph.nodes[v]['embedding']
        
        similarity = featurizer.cosine_similarity(vec_u, vec_v)
        
        # Add a small epsilon to avoid zero weights (PageRank needs positive weights)
        weight = max(similarity, 0.01)
        
        graph[u][v]['weight'] = weight
    
    print("[Ranker] All edges weighted.")
    
    print("[Ranker] Step 3: Running PageRank...")
    try:
        pagerank_scores = nx.pagerank(graph, weight='weight', alpha=0.85, max_iter=100)
        print("[Ranker] PageRank completed successfully.")
    except Exception as e:
        print(f"[Ranker] ERROR: PageRank failed: {e}")
        print("[Ranker] Falling back to unweighted PageRank...")
        try:
            pagerank_scores = nx.pagerank(graph, alpha=0.85, max_iter=100)
        except Exception as e2:
            print(f"[Ranker] ERROR: Unweighted PageRank also failed: {e2}")
            pagerank_scores = {node: 1.0 / len(graph.nodes) for node in graph.nodes}
    
    
    sorted_scores = dict(sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True))
    
    print(f"[Ranker] Ranking complete. Top 5 nodes:")
    for i, (node, score) in enumerate(list(sorted_scores.items())[:5], 1):
        print(f"  {i}. {node}: {score:.6f}")
    
    return sorted_scores


# --- Main execution block for testing ---
if __name__ == "__main__":
    print("--- Running Ranker Test ---\n")
    
    # Create a test graph
    G = nx.Graph()
    
    # Add nodes with context
    nodes_data = {
        "Nvidia": "Nvidia is the leader in AI chips with H100 and H200 GPUs. They dominate the datacenter AI market.",
        "AMD": "AMD competes with Nvidia using MI300X chips. They offer competitive pricing.",
        "Intel": "Intel is developing Gaudi accelerators for AI workloads but lags behind.",
        "Microsoft": "Microsoft Azure is a major cloud provider that uses Nvidia GPUs for AI services.",
        "OpenAI": "OpenAI trains large language models using Nvidia's GPU infrastructure.",
    }
    
    for node, context in nodes_data.items():
        G.add_node(node, combined_context=context)
    
    # Add edges (relationships)
    edges = [
        ("Nvidia", "AMD"),      # Competitors
        ("Nvidia", "Intel"),    # Competitors
        ("Nvidia", "Microsoft"), # Partnership
        ("Nvidia", "OpenAI"),   # Partnership
        ("AMD", "Intel"),       # Competitors
        ("Microsoft", "OpenAI"), # Partnership
    ]
    
    G.add_edges_from(edges)
    
    print(f"Test graph: {len(G.nodes)} nodes, {len(G.edges)} edges\n")
    
    # Run ranker
    ranked_nodes = run(G)
    
    print("\n--- TEST RESULT ---")
    print("Full Rankings:")
    for i, (node, score) in enumerate(ranked_nodes.items(), 1):
        print(f"{i}. {node}: {score:.6f}")

