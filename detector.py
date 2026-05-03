import os
import json
from ultralytics import YOLO
import cv2

def detect_objects_in_frames(frames_dir, output_json="frame_timeline.json"):
    """
    Scans extracted frames, detects objects using YOLO, 
    and builds a timeline of what appears in the video.
    """
    print("🚀 Loading YOLO11 Nano model...")
    # The 'n' stands for nano. It's extremely fast and will download automatically.
    model = YOLO("yolo11n.pt") 

    frame_timeline = {}
    
    # Create a folder to save the visual proofs
    visuals_dir = "annotated_frames"
    if not os.path.exists(visuals_dir):
        os.makedirs(visuals_dir)

    # Get all frames and sort them chronologically
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg'))])
    
    if not frames:
        print(f"⚠️ No images found in {frames_dir}. Did you run pipeline.py first?")
        return

    print(f"🔍 Analyzing {len(frames)} frames...")

    for filename in frames:
        filepath = os.path.join(frames_dir, filename)
        
        # Run YOLO inference
        results = model(filepath, verbose=False)
        
        detected_objects = []
        
        for r in results:
            # 1. Save the annotated image (for you to look at)
            annotated_frame = r.plot() 
            cv2.imwrite(os.path.join(visuals_dir, filename), annotated_frame)
            
            # 2. Extract the text data (for the pipeline)
            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                # Only keep predictions the AI is somewhat confident about
                if confidence > 0.40: 
                    detected_objects.append(class_name)
        
        # Remove duplicates (e.g., if it sees two bowls, just record 'bowl' once for the LLM)
        unique_objects = list(set(detected_objects))
        frame_timeline[filename] = unique_objects
        
        print(f"Processed {filename}: Found {unique_objects}")

    # Save our data bridge
    with open(output_json, 'w') as f:
        json.dump(frame_timeline, f, indent=4)
        
    print(f"\n✅ Done! Visuals saved to '{visuals_dir}'")
    print(f"📊 Timeline data saved to '{output_json}'")

# --- Run the code ---
if __name__ == "__main__":
    FRAMES_FOLDER = "extracted_frames"
    detect_objects_in_frames(FRAMES_FOLDER)