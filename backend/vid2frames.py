import cv2
import os
import glob
import argparse
from datetime import datetime

def extract_frames(video_path, output_dir="frames", interval=1):
    """
    Extract frames from a video file at specified intervals
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Base directory to save frames
        interval (int): Interval in seconds between frames
    
    Returns:
        str: Path to the created output directory
    """
    # Check if video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create unique output directory
    base_dir = output_dir
    existing_dirs = glob.glob(f"{base_dir}_*")
    next_num = 1
    
    if existing_dirs:
        # Extract numbers from existing directories
        dir_nums = [int(dir.split('_')[-1]) for dir in existing_dirs if dir.split('_')[-1].isdigit()]
        if dir_nums:
            next_num = max(dir_nums) + 1
    
    output_dir = f"{base_dir}_{next_num}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # Calculate frame step based on interval
    frame_step = int(fps * interval)
    
    frame_count = 0
    saved_count = 0
    
    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Extracting frames at {interval}-second intervals...")
    
    # Read and save frames
    while True:
        ret, frame = video.read()
        
        if not ret:
            break
        
        # Save frame at specified intervals
        if frame_count % frame_step == 0:
            saved_count += 1
            output_path = os.path.join(output_dir, f"frame_{saved_count}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"Saved frame {saved_count} at {frame_count / fps:.2f} seconds")
        
        frame_count += 1
    
    video.release()
    print(f"Extraction complete. Saved {saved_count} frames to '{output_dir}'")
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Extract frames from video at regular intervals")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--interval", type=int, default=1, help="Interval in seconds between frames (default: 1)")
    parser.add_argument("--output", default="frames", help="Base name for output directory (default: 'frames')")
    
    args = parser.parse_args()
    
    try:
        output_dir = extract_frames(args.video_path, args.output, args.interval)
        print(f"Frames successfully extracted to {output_dir}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()