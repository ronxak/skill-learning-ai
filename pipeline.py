import cv2
import os

def extract_frames(video_path, output_dir, sample_rate_seconds=1):
    """
    Extracts frames from a video at a specified interval.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the video
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video Loaded: {fps} FPS | Total Duration: {duration:.2f} seconds")
    print(f"Extracting 1 frame every {sample_rate_seconds} second(s)...")

    # Calculate how many frames to skip to match the sample rate
    frame_interval = fps * sample_rate_seconds
    
    current_frame = 0
    saved_count = 0

    while True:
        success, frame = video.read()
        if not success:
            break # Video finished

        # Save frame if it matches our interval
        if current_frame % frame_interval == 0:
            timestamp = current_frame // fps
            filename = os.path.join(output_dir, f"frame_{timestamp:04d}s.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        current_frame += 1

    video.release()
    print(f"Success: Extracted {saved_count} frames to '{output_dir}'")

# --- Test the code ---
if __name__ == "__main__":
    # TODO: Put a short test video in the same folder and update this name
    TEST_VIDEO = "test_video.mp4" 
    OUTPUT_FOLDER = "extracted_frames"
    
    extract_frames(TEST_VIDEO, OUTPUT_FOLDER, sample_rate_seconds=2)