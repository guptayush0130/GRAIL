"""
AGORA v2: Graph-Backed Question Answering System
Main entry point that orchestrates all components to produce AGORA_Response.
"""

import networkx as nx
from dataclasses import dataclass
from typing import Dict
import json

import entity_extractor
import graph_builder
import ranker
import synthesizer


@dataclass
class AGORA_Response:
    """
    The output format for AGORA v2 system.
    """
    final_answer: str
    ranked_nodes: Dict[str, float]
    graph: nx.Graph
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "final_answer": self.final_answer,
            "ranked_nodes": self.ranked_nodes,
            "graph_stats": {
                "num_nodes": len(self.graph.nodes),
                "num_edges": len(self.graph.edges),
                "nodes": list(self.graph.nodes)
            }
        }
    
    def __str__(self):
        """Pretty string representation"""
        result = "=" * 80 + "\n"
        result += "AGORA v2 RESPONSE\n"
        result += "=" * 80 + "\n\n"
        
        result += "FINAL ANSWER:\n"
        result += "-" * 80 + "\n"
        result += self.final_answer + "\n\n"
        
        result += "TOP RANKED NODES:\n"
        result += "-" * 80 + "\n"
        for i, (node, score) in enumerate(list(self.ranked_nodes.items())[:10], 1):
            result += f"{i}. {node}: {score:.6f}\n"
        
        result += "\n"
        result += "GRAPH STATISTICS:\n"
        result += "-" * 80 + "\n"
        result += f"Total Nodes: {len(self.graph.nodes)}\n"
        result += f"Total Edges: {len(self.graph.edges)}\n"
        result += f"Average Degree: {sum(dict(self.graph.degree()).values()) / len(self.graph.nodes):.2f}\n"
        
        result += "\n" + "=" * 80 + "\n"
        return result


def run(user_query: str, max_nodes: int = 25, max_depth: int = 2, top_n_for_answer: int = 5) -> AGORA_Response:
    """
    The main AGORA v2 pipeline.
    
    Transforms a user question into a synthesized, graph-backed answer through 7 stages:
    1. Query Input
    2. Entity Extraction (LLM)
    3. Graph Building (BFS with Text + Video Sense)
    4. Entity Expansion (LLM)
    5. Graph Featurization (SBERT)
    6. Ranking (GRAIL PageRank)
    7. Answer Synthesis (LLM)
    
    Args:
        user_query: The user's question
        max_nodes: Maximum number of nodes in the graph (default: 25)
        max_depth: Maximum depth for BFS (default: 2)
        top_n_for_answer: Number of top nodes to use for synthesis (default: 5)
        
    Returns:
        AGORA_Response object containing final_answer, ranked_nodes, and graph
    """
    print("\n" + "=" * 80)
    print("AGORA v2 - Graph-Backed Question Answering System")
    print("=" * 80 + "\n")
    
    print(f"User Query: {user_query}\n")
    
    # Stage 1: Query Input (already received)
    print("[Stage 1/7] Query received ✓\n")
    
    # Stage 2: Entity Extraction
    print("[Stage 2/7] Extracting seed entity...")
    seed_entity = entity_extractor.get_seed_entity(user_query)
    
    if not seed_entity:
        return AGORA_Response(
            final_answer="Error: Could not extract seed entity from query.",
            ranked_nodes={},
            graph=nx.Graph()
        )
    
    print(f"[Stage 2/7] Seed entity: '{seed_entity}' ✓\n")
    
    # Stage 3-4: Graph Building (with Text Sense, Video Sense, and Entity Expansion)
    print("[Stage 3-4/7] Building knowledge graph...")
    builder = graph_builder.GraphBuilder(max_nodes=max_nodes, max_depth=max_depth)
    graph = builder.build(seed_entity, user_query)
    
    if len(graph.nodes) == 0:
        return AGORA_Response(
            final_answer="Error: Failed to build knowledge graph.",
            ranked_nodes={},
            graph=graph
        )
    
    print(f"[Stage 3-4/7] Graph built: {len(graph.nodes)} nodes, {len(graph.edges)} edges ✓\n")
    
    # Stage 5-6: Ranking (Featurization + PageRank)
    print("[Stage 5-6/7] Running GRAIL ranking algorithm...")
    ranked_nodes = ranker.run(graph)
    
    if not ranked_nodes:
        return AGORA_Response(
            final_answer="Error: Failed to rank nodes.",
            ranked_nodes={},
            graph=graph
        )
    
    print(f"[Stage 5-6/7] Ranking complete ✓\n")
    
    # Stage 7: Answer Synthesis
    print("[Stage 7/7] Synthesizing final answer...")
    final_answer = synthesizer.generate(user_query, ranked_nodes, graph, top_n=top_n_for_answer)
    print(f"[Stage 7/7] Answer synthesized ✓\n")
    
    print("=" * 80)
    print("AGORA v2 COMPLETE")
    print("=" * 80 + "\n")
    
    return AGORA_Response(
        final_answer=final_answer,
        ranked_nodes=ranked_nodes,
        graph=graph
    )


# --- Main execution block for testing ---
if __name__ == "__main__":
    
    # Test query
    test_query = "Give me the latest products, performance, competitors, opportunities, and key collaborators of Nvidia. What things are going to affect it the most?"
    
    print(f"Test Query:\n{test_query}\n")
    print("-" * 80 + "\n")
    
    # Run AGORA (with smaller limits for testing)
    response = run(
        user_query=test_query,
        max_nodes=10,  # Smaller for testing
        max_depth=1,   # Smaller for testing
        top_n_for_answer=5
    )
    
    # Display results
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80 + "\n")
    
    print(response)
    
    # Save to file
    output_file = "agora_response.json"
    with open(output_file, 'w') as f:
        json.dump(response.to_dict(), f, indent=2)
    
    print(f"\nResponse also saved to: {output_file}")

