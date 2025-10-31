from openai import OpenAI
import json

try:
    from config import OPENAI_API_KEY
except ImportError:
    print("ERROR: config.py not found or OPENAI_API_KEY is missing.")
    exit()

client = OpenAI(api_key=OPENAI_API_KEY)

def get_seed_entity(user_query: str) -> str:
    print(f"[EntityExtractor] Extracting seed entity from query...")
    
    prompt = f"""You are an analyzer AI. Your task is to extract the single, primary entity from the user's query.

The seed entity should be:
- A company, person, product, or organization
- The main subject of the query
- Specific and unambiguous

Return ONLY the entity name, nothing else. No explanations, no punctuation.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"User Query: {user_query}"}
            ],
            temperature=0.0,
        )
        
        seed_entity = response.choices[0].message.content.strip()
        print(f"[EntityExtractor] Extracted seed entity: '{seed_entity}'")
        return seed_entity
        
    except Exception as e:
        print(f"[EntityExtractor] ERROR: Failed to extract seed entity: {e}")
        return None


# --- Main execution block for testing ---
if __name__ == "__main__":
    print("--- Running Entity Extractor Test ---\n")
    
    test_queries = [
        "Give me the latest products, performance, competitors, opportunities, and key collaborators of Nvidia. What things are going to affect it the most?",
        "What are Apple's latest innovations in AI and who are they competing with?",
        "Tell me about Tesla's autonomous driving technology and its partnerships"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        entity = get_seed_entity(query)
        print(f"Result: {entity}")
        print("-" * 80)

