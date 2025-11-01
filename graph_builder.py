import networkx as nx
from queue import Queue
from typing import Tuple

import text_sense
import janus
import entity_sense


class GraphBuilder:
    def __init__(self, max_nodes: int = 25, max_depth: int = 2):
        """
        Initialize the GraphBuilder.
        
        Args:
            max_nodes: Maximum number of nodes to add to the graph
            max_depth: Maximum depth for breadth-first search
        """
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.graph = nx.Graph()
        self.processed_nodes = set()
        self.queue = Queue()
    
    def build(self, seed_entity: str, user_query: str) -> nx.Graph:
        
        print(f"[GraphBuilder] Starting graph construction...")
        print(f"[GraphBuilder] Seed: '{seed_entity}', Max Nodes: {self.max_nodes}, Max Depth: {self.max_depth}")
        
        self.queue.put((seed_entity, 0))
        
        while not self.queue.empty() and len(self.processed_nodes) < self.max_nodes:
            current_entity, current_depth = self.queue.get()
            
            if current_entity in self.processed_nodes:
                continue
            
            if current_depth > self.max_depth:
                print(f"[GraphBuilder] Max depth reached for '{current_entity}' (depth {current_depth})")
                continue
            
            print(f"\n[GraphBuilder] Processing: '{current_entity}' (Depth: {current_depth}, Graph Size: {len(self.graph.nodes)})")
            
            self.processed_nodes.add(current_entity)
            
            combined_context = self._ingest_data(current_entity, user_query)
            
            if not combined_context:
                print(f"[GraphBuilder] WARNING: No data obtained for '{current_entity}', skipping...")
                continue
            
            self.graph.add_node(current_entity, combined_context=combined_context, depth=current_depth)
            print(f"[GraphBuilder] Added node '{current_entity}' to graph (Total: {len(self.graph.nodes)})")
            
            if len(self.graph.nodes) >= self.max_nodes:
                print(f"[GraphBuilder] Reached max nodes ({self.max_nodes}), stopping expansion.")
                break
            
            if current_depth < self.max_depth:
                new_entities = entity_sense.get_entities(combined_context, user_query, current_entity)
                
                if new_entities:
                    print(f"[GraphBuilder] Found {len(new_entities)} new entities to explore")
                    
                    for new_entity in new_entities:
                        if new_entity not in self.graph.nodes:
                            self.graph.add_node(new_entity, combined_context="", depth=current_depth + 1)
                        
                        if not self.graph.has_edge(current_entity, new_entity):
                            self.graph.add_edge(current_entity, new_entity)
                            print(f"[GraphBuilder] Added edge: {current_entity} -> {new_entity}")
                        
                        if new_entity not in self.processed_nodes:
                            self.queue.put((new_entity, current_depth + 1))
                else:
                    print(f"[GraphBuilder] No new entities found for '{current_entity}'")
            else:
                print(f"[GraphBuilder] Not expanding '{current_entity}' (at max depth)")
        
        print(self.queue.qsize())
        print(f"\n[GraphBuilder] Graph construction complete!")
        print(f"[GraphBuilder] Final Stats: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
        
        return self.graph
    
    def _ingest_data(self, entity: str, user_query: str) -> str:
        
        print(f"[GraphBuilder] Ingesting data for '{entity}'...")
        
        # Step 4a: Text Sense (Parallel API)
        try:
            initial_answer, event_names = text_sense.get_text_and_events(entity, user_query)
            print(f"[GraphBuilder] Text Sense completed for '{entity}'")
        except Exception as e:
            print(f"[GraphBuilder] ERROR: Text Sense failed for '{entity}': {e}")
            initial_answer = ""
            event_names = ""
        
        # Convert answer to string if it's a dict/object
        if isinstance(initial_answer, dict):
            initial_answer = str(initial_answer)
        elif initial_answer is None:
            initial_answer = ""
        
        # Step 4b: Video Sense (Janus Module)
        video_transcripts = ""
        if event_names:
            try:
                # Convert event_names to list if it's a string
                if isinstance(event_names, str):
                    events_list = [event_names]
                elif isinstance(event_names, list):
                    events_list = event_names
                else:
                    events_list = [str(event_names)]
                
                print(f"[GraphBuilder] Searching videos for events: {events_list}")
                video_transcripts = janus.get_transcripts(events_list, user_query)
                print(f"[GraphBuilder] Video Sense completed for '{entity}' ({len(video_transcripts)} chars)")
            except Exception as e:
                print(f"[GraphBuilder] ERROR: Video Sense failed for '{entity}': {e}")
                video_transcripts = ""
        else:
            print(f"[GraphBuilder] No events found, skipping Video Sense for '{entity}'")
        
        # Combine contexts
        combined = f"{initial_answer}\n\n{video_transcripts}".strip()
        
        print(f"[GraphBuilder] Combined context for '{entity}': {len(combined)} characters")
        
        return combined


# --- Main execution block for testing ---
if __name__ == "__main__":
    print("--- Running GraphBuilder Test ---\n")
    
    test_seed = "Nvidia"
    test_query = "Give me the latest products, performance, competitors, opportunities, and key collaborators of Nvidia."
    
    # Build the graph (using smaller limits for testing)
    builder = GraphBuilder(max_nodes=10, max_depth=1)
    graph = builder.build(test_seed, test_query)
    
    print("\n--- TEST RESULT ---")
    print(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    print("\nNodes:")
    for node in graph.nodes:
        context_length = len(graph.nodes[node].get('combined_context', ''))
        depth = graph.nodes[node].get('depth', '?')
        print(f"  - {node} (Depth: {depth}, Context: {context_length} chars)")
    
    print("\nEdges:")
    for u, v in graph.edges:
        print(f"  - {u} <-> {v}")

