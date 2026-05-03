import os
import json
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

def generate_video_steps(json_path="frame_timeline.json"):
    """
    Reads the YOLO timeline data and uses Groq (Llama-3) to reason 
    about the actions and generate step-by-step instructions.
    """
    # --- SECURE API KEY LOAD ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    if not GROQ_API_KEY:
        print("❌ Error: GROQ_API_KEY not found.")
        print("Make sure you have a .env file in this directory with GROQ_API_KEY=your_key")
        return
    
    # Initialize the Groq client
    client = Groq(api_key=GROQ_API_KEY)

    print("📖 Reading timeline data...")
    try:
        with open(json_path, 'r') as f:
            timeline_data = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Could not find {json_path}. Did you run detector.py?")
        return

    # Convert the JSON dictionary into a clean string for the prompt
    timeline_str = json.dumps(timeline_data, indent=2)

    # --- THE SYSTEM PROMPT ---
    system_prompt = """
    You are an advanced Video Action Analysis AI. 
    Your job is to read a chronological timeline of objects detected in a video and figure out what the person is doing.
    
    Rules:
    1. Ignore random object detection errors or noise (e.g., if a potted plant randomly appears for 1 frame, ignore it).
    2. Deduce the overarching activity.
    3. Break the activity down into 3 to 5 logical, sequential steps.
    
    Format your output EXACTLY like this:
    **Overall Action:** [Short description]
    
    **Steps:**
    Step 1: [Action] 
    Step 2: [Action] 
    """

    print("⚡ Sending data to Groq (Llama 3 70B). Prepare for speed...")
    
    try:
        # Make the API call to Groq
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": f"Here is the timeline data:\n{timeline_str}",
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct", 
            temperature=0.2, 
        )
        
        print("\n" + "="*50)
        print("✨ AI GENERATED STEPS ✨")
        print("="*50)
        print(chat_completion.choices[0].message.content)
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"❌ An error occurred with the Groq API: {e}")

# --- Run the code ---
if __name__ == "__main__":
    generate_video_steps()