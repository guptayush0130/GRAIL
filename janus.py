from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
from openai import OpenAI
import json
import os
import time
import ssl

try:
    from config import YOUTUBE_API_KEY, OPENAI_API_KEY
except ImportError:
    print("ERROR: config.py not found or API keys are missing.")
    exit()

try:
    youtube_service = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"Error initializing API clients: {e}")
    exit()

def summarize_transcript(transcript: str, user_query: str, event_name: str) -> str:
    
    if not transcript or len(transcript.strip()) == 0:
        return ""
    
    print(f"[Janus] Summarizing transcript using LLM (length: {len(transcript)} chars)...")
    
    prompt = f"""You are a research analyst. Extract and summarize ONLY the information from the video transcript that is relevant to answering the user's query.

Instructions:
- Extract key facts, announcements, and details that relate to the user's query
- Focus on concrete information (products, numbers, partnerships, dates, etc.)
- Ignore irrelevant content, introductions, and off-topic discussions
- Keep the summary concise but informative (200-500 words)
- If the transcript contains nothing relevant to the query, return "No relevant information found"

Provide your focused summary:"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", f"content": "User Query: {user_query} \n Video Event: {event_name} \n Video Transcript: {transcript}"}
            ],
            temperature=0.3,
        )
        
        summary = response.choices[0].message.content.strip()
        print(f"[Janus] Summary generated ({len(summary)} chars)")
        return summary
        
    except Exception as e:
        print(f"[Janus] ERROR: Failed to summarize transcript: {e}")
        return transcript

def find_video_by_exact_query(query: str) -> str | None:
    try:
        # Search for the query, limit to 1 top result
        search_response = youtube_service.search().list(
            q=query,
            part='snippet',
            maxResults=1, 
            type='video',
            order='relevance',
            videoDuration='long'
        ).execute()
        
        # Check if the 'result' key exists and has at least one item
        if not search_response.get('items'):
            print(f"[Janus] ERROR: No video found for exact query: {query}")
            return None

        video_id = search_response['items'][0]['id']['videoId']
        video_title = search_response['items'][0]['snippet']['title']
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        print(f"[Janus] Found video: '{video_title}'")
        print(f"[Janus] Video title: {video_title}")
        print(f"[Janus] Video URL: {video_url}")

        return video_id
    except Exception as e:
        print(f"[Janus] ERROR: Error during YouTube search for '{query}': {e}")
        return None

def transcribe_video(video_id: str) -> str | None:
    try:
        # --- Method 1: Try the fast, free API first ---
        print(f"[Janus] Attempting fast transcription for video {video_id}...")
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id)
        
        # Combine all 'text' parts into one large string
        full_transcript = " ".join([chunk.text for chunk in transcript_list])
        
        # Clean up newlines and extra spaces
        full_transcript = " ".join(full_transcript.replace("\n", " ").split())
        print(f"[Janus] ...Fast transcription successful.")
        print(f"[Janus] ...Full transcript: {full_transcript}")
        return full_transcript
        
    except Exception as e:
        # --- Method 2: Fallback to OpenAI Whisper ---
        print(f"[Janus] WARN: Fast transcription failed ({e}).")
        if not openai_client:
            print("[Janus] ERROR: Whisper fallback failed because OpenAI client is not initialized.")
            return None
            
        print("[Janus] Attempting Whisper fallback...")

        try:
            ssl._create_default_https_context = ssl._create_unverified_context
            print("[Janus] INFO: Applied SSL context workaround for pytube.")
        except Exception as ssl_e:
            print(f"[Janus] WARN: Could not apply SSL context workaround: {ssl_e}")

        video_url = f"https://www.youtube.com/watch?v={video_id}"
        temp_audio_file = f"temp_audio_{video_id}.m4a"

        try:
            # 1. Download audio using pytube
            print(f"[Janus] ...Downloading audio for {video_id}...")
            yt = YouTube(video_url)
            audio_stream = yt.streams.get_audio_only()
            if not audio_stream:
                print(f"[Janus] ERROR: No audio stream found for {video_id}.")
                return None
            
            audio_stream.download(filename=temp_audio_file)
            print(f"[Janus] ...Audio downloaded to {temp_audio_file}.")

            # 2. Transcribe with Whisper
            print(f"[Janus] ...Sending audio to Whisper API for transcription...")
            with open(temp_audio_file, "rb") as audio_file:
                transcript_response = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            full_transcript = transcript_response.text
            print(f"[Janus] ...Whisper transcription successful.")
            return full_transcript

        except Exception as whisper_e:
            print(f"[Janus] ERROR: Whisper fallback failed: {whisper_e}")
            return None
        finally:
            # 3. Clean up the temporary audio file
            if os.path.exists(temp_audio_file):
                os.remove(temp_audio_file)
                print(f"[Janus] ...Cleaned up temporary file {temp_audio_file}.")


def get_transcripts(event_names: list[str], user_query: str) -> str:
    
    all_summaries = []
    
    if not event_names:
        print("[Janus] No event names provided to search.")
        return ""
        
    print(f"[Janus] Processing {len(event_names)} events...")
    
    for event in event_names:
        print(f"[Janus] --- Processing Event: '{event}' ---")
        
        # 1. Find the video
        video_id = find_video_by_exact_query(event)
        
        if video_id:
            print(f"[Janus] Found video ID: {video_id} (https://www.youtube.com/watch?v={video_id})")
            
            # 2. Transcribe the video
            transcript_text = transcribe_video(video_id)
            
            if transcript_text:
                print(f"[Janus] Successfully transcribed video ({len(transcript_text)} chars)")
                
                # 3. Summarize transcript based on user query
                summary = summarize_transcript(transcript_text, user_query, event)
                
                if summary and summary != "No relevant information found":
                    all_summaries.append(f"[{event}] {summary}")
                    print(f"[Janus] Added summary for '{event}'")
                else:
                    print(f"[Janus] No relevant information in transcript for '{event}'")
            else:
                print(f"[Janus] Transcription failed for this video")
        else:
            print(f"[Janus] No relevant video found for '{event}'")
            
    print(f"[Janus] --- Finished processing all events ({len(all_summaries)} summaries generated)")
    
    # Combine all summaries
    return "\n\n".join(all_summaries)

# --- Main execution block for testing ---
if __name__ == "__main__":
    print("--- Running Janus Module Test ---")
    
    test_query = "What are the latest products and announcements from Nvidia?"
    
    test_events_list = [
        "Nvidia GTC 2025 Keynote"
    ]

    print(f"\nTest Query: {test_query}")
    print(f"Test Events: {test_events_list}\n")
    
    summaries = get_transcripts(test_events_list, test_query)
    
    if summaries:
        print("\n--- TEST RESULT (SUCCESS) ---")
        print(f"Total length: {len(summaries)} chars.")
        print("\nSummaries:")
        print(summaries)
    else:
        print("\n--- TEST RESULT (FAILED) ---")
        print("No summaries generated.")