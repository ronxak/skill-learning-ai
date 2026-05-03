import gradio as gr
import cv2
import os
import json
from ultralytics import YOLO
from groq import Groq
from dotenv import load_dotenv

# Load API Key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load YOLO model once at startup to save time
print("Loading YOLO model...")
yolo_model = YOLO("yolo11n.pt") 

def analyze_video(video_filepath):
    """
    The master function that runs the entire pipeline from video upload to text output.
    """
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY not found in .env file."
        
    client = Groq(api_key=GROQ_API_KEY)
    
    # --- PHASE 1: FRAME EXTRACTION ---
    yield "Step 1: Extracting frames from video..."
    
    video = cv2.VideoCapture(video_filepath)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30 # fallback
    
    frame_interval = fps * 2 # 1 frame every 2 seconds
    current_frame = 0
    extracted_frames = []
    
    while True:
        success, frame = video.read()
        if not success:
            break
        if current_frame % frame_interval == 0:
            extracted_frames.append(frame)
        current_frame += 1
    video.release()
    
    # --- PHASE 2: OBJECT DETECTION (YOLO) ---
    yield f"Step 2: Analyzing {len(extracted_frames)} frames with YOLO Vision..."
    
    timeline_data = {}
    for i, frame in enumerate(extracted_frames):
        # Run YOLO on the frame array directly (no need to save to disk!)
        results = yolo_model(frame, verbose=False)
        detected_objects = []
        
        for r in results:
            for box in r.boxes:
                if float(box.conf[0]) > 0.40: # Confidence threshold
                    class_name = yolo_model.names[int(box.cls[0])]
                    detected_objects.append(class_name)
                    
        timeline_data[f"Frame_{i+1}"] = list(set(detected_objects))
        
    timeline_str = json.dumps(timeline_data, indent=2)

    # --- PHASE 3: TEMPORAL REASONING (GROQ) ---
    yield "Step 3: Groq meta-llama is reasoning over the timeline data..."
    
    system_prompt = """
    You are an advanced Video Action Analysis AI. 
    Read this chronological timeline of objects detected in a video.
    
    1. Ignore random object detection errors (noise).
    2. Deduce the overarching activity.
    3. Break the activity down into logical, sequential steps.
    
    Format your output EXACTLY like this template:
    
    **Overall Action:** [Short description]
    
    **Steps:**
    1. **[Step Name]:** [Step action description]
    <br>*Explanation:* [How you deduced this]
    
    2. **[Step Name]:** [Step action description]
    <br>*Explanation:* [How you deduced this]
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Timeline data:\n{timeline_str}"}
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct", 
            temperature=0.2, 
        )
        final_output = chat_completion.choices[0].message.content
        
        # We append the raw JSON at the bottom so recruiters can see the "X-Ray" data
        debug_info = f"\n\n---\n### Developer X-Ray (Raw YOLO Timeline)\n```json\n{timeline_str}\n```"
        
        yield final_output + debug_info
        
    except Exception as e:
        yield f"❌ Groq API Error: {str(e)}"

# --- BUILD THE UI ---

# Custom CSS for centering and forcing the subtitle onto one line
custom_css = """
#centered-container {
    max-width: 900px; 
    margin: 0 auto; 
}
.nowrap-subtitle {
    font-size: 1.1em; 
    color: #a0a0a0; 
    white-space: nowrap; 
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    
    gr.HTML("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="font-size: 2.5em; font-weight: bold; margin-bottom: 10px;"> Skill Learning from Video Pipeline</h1>
            <p class="nowrap-subtitle">
                Upload a video. This system uses <b>OpenCV</b> for frame extraction, <b>YOLO11</b> for spatial object detection, and <b>Groq (meta-llama)</b> for temporal reasoning to deduce the steps.
            </p>
        </div>
    """)
    
    with gr.Column(elem_id="centered-container"):
        video_input = gr.Video(label="Upload Video (e.g., cooking, gym)")
        submit_btn = gr.Button("Generate Steps", variant="primary", size="lg")
        
        gr.HTML("<hr style='margin-top: 20px; margin-bottom: 20px;'/>")
        
        text_output = gr.Markdown(label="AI Reasoning Output")
            
    submit_btn.click(fn=analyze_video, inputs=video_input, outputs=text_output)

if __name__ == "__main__":
    demo.launch(share=True)