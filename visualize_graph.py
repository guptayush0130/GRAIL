"""
Graph Visualization Utility for AGORA v2

Visualizes the knowledge graph produced by AGORA v2.
"""

import networkx as nx
import matplotlib.pyplot as plt
from agora import run


def visualize_graph(graph: nx.Graph, title: str = "AGORA v2 Knowledge Graph", save_file: str = None):
    """
    Visualize a NetworkX graph with better layout and styling.
    
    Args:
        graph: NetworkX graph to visualize
        title: Title for the plot
        save_file: If provided, saves the plot to this file
    """
    plt.figure(figsize=(16, 12))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        graph, pos,
        node_color='lightblue',
        node_size=3000,
        alpha=0.9,
        edgecolors='black',
        linewidths=2
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        graph, pos,
        edge_color='gray',
        width=2,
        alpha=0.6
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        graph, pos,
        font_size=10,
        font_weight='bold',
        font_family='sans-serif'
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Graph saved to: {save_file}")
    
    plt.show()


def visualize_with_rankings(graph: nx.Graph, ranked_nodes: dict, title: str = "AGORA v2 Graph with Rankings", save_file: str = None):
    """
    Visualize graph with node sizes proportional to PageRank scores.
    
    Args:
        graph: NetworkX graph to visualize
        ranked_nodes: Dictionary of {node: pagerank_score}
        title: Title for the plot
        save_file: If provided, saves the plot to this file
    """
    plt.figure(figsize=(16, 12))
    
    # Use spring layout
    pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
    
    # Get node sizes based on PageRank (normalized)
    max_score = max(ranked_nodes.values())
    node_sizes = [ranked_nodes.get(node, 0) / max_score * 5000 for node in graph.nodes()]
    
    # Get node colors based on rank
    node_colors = []
    for node in graph.nodes():
        rank = list(ranked_nodes.keys()).index(node) + 1 if node in ranked_nodes else len(graph.nodes())
        if rank <= 3:
            node_colors.append('gold')
        elif rank <= 5:
            node_colors.append('lightblue')
        else:
            node_colors.append('lightgray')
    
    # Draw nodes
    nx.draw_networkx_nodes(
        graph, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        edgecolors='black',
        linewidths=2
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        graph, pos,
        edge_color='gray',
        width=2,
        alpha=0.6
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        graph, pos,
        font_size=9,
        font_weight='bold',
        font_family='sans-serif'
    )
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gold', edgecolor='black', label='Top 3'),
        Patch(facecolor='lightblue', edgecolor='black', label='Top 4-5'),
        Patch(facecolor='lightgray', edgecolor='black', label='Others')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Graph saved to: {save_file}")
    
    plt.show()


# --- Example Usage ---
if __name__ == "__main__":
    print("\n" + "="*80)
    print("AGORA v2 Graph Visualization")
    print("="*80 + "\n")
    
    # Run a query
    query = "What are Apple's latest AI developments?"
    
    print(f"Running query: {query}\n")
    response = run(query, max_nodes=10, max_depth=2)
    print(response.final_answer)
    
    print("\n" + "="*80)
    print("Visualizing Graph...")
    print("="*80 + "\n")
    
    # Basic visualization
    visualize_graph(
        response.graph,
        title=f"Knowledge Graph for: '{query}'",
        save_file="graph_basic.png"
    )
    
    # Visualization with rankings
    visualize_with_rankings(
        response.graph,
        response.ranked_nodes,
        title=f"Ranked Knowledge Graph for: '{query}'",
        save_file="graph_ranked.png"
    )
