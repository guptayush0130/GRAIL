"""
Example Usage of AGORA v2 System

This script demonstrates how to use the AGORA v2 system to answer complex questions.
"""

from agora import run
import json


def example_1():
    """Example 1: Tech company analysis"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Tech Company Analysis")
    print("="*80 + "\n")
    
    query = "Give me the latest products, performance, competitors, opportunities, and key collaborators of Nvidia. What things are going to affect it the most?"
    
    response = run(
        user_query=query,
        max_nodes=15,      # Moderate graph size
        max_depth=2,       # Two levels of connections
        top_n_for_answer=5 # Use top 5 nodes for answer
    )
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")
    print(response)
    
    return response


def example_2():
    """Example 2: Quick query with smaller graph"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Quick Query (Smaller Graph)")
    print("="*80 + "\n")
    
    query = "What are Apple's latest AI developments?"
    
    response = run(
        user_query=query,
        max_nodes=8,       # Smaller graph for faster results
        max_depth=1,       # Only direct connections
        top_n_for_answer=3 # Use top 3 nodes
    )
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")
    print(response)
    
    return response


def example_3():
    """Example 3: Deep analysis with larger graph"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Deep Analysis (Larger Graph)")
    print("="*80 + "\n")
    
    query = "What is the competitive landscape in AI chips and who are the key players?"
    
    response = run(
        user_query=query,
        max_nodes=25,      # Full graph size
        max_depth=2,       # Deep exploration
        top_n_for_answer=7 # Use more nodes for comprehensive answer
    )
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")
    print(response)
    
    return response


def save_response_to_file(response, filename="response.json"):
    """Save response to JSON file"""
    with open(filename, 'w') as f:
        json.dump(response.to_dict(), f, indent=2)
    print(f"\nResponse saved to: {filename}")


if __name__ == "__main__":
    print("\n" + "🚀"*40)
    print("AGORA v2 - Example Usage")
    print("🚀"*40 + "\n")
    
    # Run examples
    # Uncomment the examples you want to run
    
    # Example 1: Full analysis (will take 3-5 minutes)
    # response = example_1()
    # save_response_to_file(response, "nvidia_analysis.json")
    
    # Example 2: Quick query (will take 1-2 minutes)
    response = example_2()
    save_response_to_file(response, "apple_quick.json")
    
    # Example 3: Deep analysis (will take 5-7 minutes)
    # response = example_3()
    # save_response_to_file(response, "ai_chips_deep.json")


