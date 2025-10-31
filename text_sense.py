import requests
import json
from parallel import Parallel

try:
    from config import PARALLEL_API_KEY
except ImportError:
    print("ERROR: config.py not found or PARALLEL_API_KEY is missing.")
    print("Please create a config.py file and add: PARALLEL_API_KEY = 'your_key_here'")
    exit()

client = Parallel(api_key=PARALLEL_API_KEY)

def get_text_and_events(entity: str, user_query: str):
    
    headers = {
        "x-api-key": PARALLEL_API_KEY,
        "Content-Type": "application/json"
    }
    url_suggest = "https://api.parallel.ai/v1beta/tasks/suggest"

    data_answer = {
        "user_intent": f"A concise answer to {user_query} as it relates to {entity}"
    }
    
    print(f"[TextSense] Suggesting schema for answer...")
    response_answer = requests.post(url_suggest, headers=headers, json=data_answer)
    result_answer = response_answer.json()
    output_schema_answer = result_answer.get("output_schema", {})

    print(f"[TextSense] Output schema for answer: {output_schema_answer}")

    print(f"[TextSense] Running task for answer...")
    task_run_answer = client.task_run.create(
            input=f"{entity}",
            task_spec={
              "output_schema": {
                  "type": "json",  
                  "json_schema": output_schema_answer 
              }
            },
            processor="core"
        )
    run_result_answer = client.task_run.result(task_run_answer.run_id, api_timeout=3600)
    concise_answer_content = run_result_answer.output.content
    print(f"[TextSense] Got answer content- {concise_answer_content}.")

    print(f"[TextSense] Running task for events...")
    task_run_events = client.task_run.create(
            input=f"Entity: {entity} \n User Query: {user_query}", 
            task_spec={
              "output_schema": "The official name of the company's recent and important keynote, press release, or major product launch with the year and month. The event should be related to the user query and the entity. Bigger and detailed events are preferred. e.g., 'Q4 2025 Earnings Call', 'Product Launch' as a string"
            },
            processor="base" 
        )

    run_result_events = client.task_run.result(task_run_events.run_id, api_timeout=3600)
    events_content = run_result_events.output.content
    print(f"[TextSense] Got events content.")
    
    answer_data = concise_answer_content
    events_data = events_content

    return answer_data, events_data

# --- Main execution block for testing ---
if __name__ == "__main__":
    print("--- Running TextSense Module Test (Suggest-based) ---")
    
    test_query = "Give me the latest products, performance, competitors, opportunities, and key collaborators of Nvidia. What things are going to affect it the most."
    test_entity = "Nvidia"
    
    answer, events = get_text_and_events(test_entity, test_query)
    
    if answer or events:
        print("\n--- TEST RESULT ---")
        print(f"\n[Entity]: {test_entity}")
        
        print("\n[Concise Answer]:")
        print(json.dumps(answer, indent=2))
        
        print("\n[Recent Key Event]:")
        print(json.dumps(events, indent=2))

        print("-------------------")
    else:
        print("\n--- TEST FAILED ---")
        print("No data was returned. Check API key and task configuration.")