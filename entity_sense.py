from openai import OpenAI
import json

try:
    from config import OPENAI_API_KEY
except ImportError:
    print("ERROR: config.py not found or OPENAI_API_KEY is missing.")
    exit()

client = OpenAI(api_key=OPENAI_API_KEY)

def get_entities(combined_context: str, user_query: str, current_entity: str) -> list[str]:
    print(f"[EntitySense] Extracting new entities from context for '{current_entity}'...")
    
    prompt = f"""You are a research analyst AI. Your task is to identify the TOP 5 most relevant new entities to investigate next based on the user's query and the context.

Based on the user's query and the context above, identify the TOP 3-5 most relevant entities (companies, products, people, or organizations) that should be investigated next to answer the user's question.

Requirements:
- Return ONLY entities that are DIFFERENT from the current entity
- Prioritize entities that are directly mentioned in the context
- Focus on entities most relevant to the user's query
- Entities should be companies, products, people, or initiatives
- Return as a valid JSON array of strings

Example response format:
["Entity1", "Entity2", "Entity3", "Entity4", "Entity5"]

The format should be exactly like the example response format, don't add any quotes or JSON tags."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"User's Original Query: {user_query} \n Current Entity Being Analyzed: {current_entity} \n Context About {current_entity}: {combined_context}"}
            ],
            temperature=0.3,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Try to parse as JSON
        try:
            entities = json.loads(content)
            if isinstance(entities, list):
                entities = [e for e in entities if e.lower() != current_entity.lower()][:5]
                print(f"[EntitySense] Extracted {len(entities)} new entities: {entities}")
                return entities
            else:
                print(f"[EntitySense] ERROR: Response is not a list: {content}")
                return []
        except json.JSONDecodeError:
            print(f"[EntitySense] ERROR: Could not parse JSON from response: {content}")
            return []
        
    except Exception as e:
        print(f"[EntitySense] ERROR: Failed to extract entities: {e}")
        return []


# --- Main execution block for testing ---
if __name__ == "__main__":
    print("--- Running EntitySense Test ---\n")
    
    test_context = """
    Nvidia has been a leader in GPU technology, with their latest H100 and H200 chips dominating
    the AI training market. Their main competitors include AMD with their MI300X chips and 
    Intel's Gaudi accelerators. Nvidia has partnerships with Microsoft Azure, Amazon AWS, and 
    Google Cloud for their cloud infrastructure. CEO Jensen Huang recently announced 
    the new Blackwell architecture at GTC 2025. The company faces competition from custom chips 
    like Google's TPU and Amazon's Trainium. They also collaborate with OpenAI, Meta, and 
    Anthropic on AI model training.
    """
    
    test_query = "Give me the latest products, performance, competitors, opportunities, and key collaborators of Nvidia"
    test_entity = "Nvidia"
    
    entities = get_entities(test_context, test_query, test_entity)
    
    print("\n--- TEST RESULT ---")
    print(f"Query: {test_query}")
    print(f"Current Entity: {test_entity}")
    print(f"\nExtracted Entities: {json.dumps(entities, indent=2)}")

