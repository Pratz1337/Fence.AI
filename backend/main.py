from io import BytesIO
from scripts.CropFaces import *
from scripts.DetectDeepfake import *
from scripts.DetectArt import *
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import io
from PIL import Image
import c2pa
import tempfile
import os
import json
import cv2
import glob
import shutil
from datetime import timedelta, datetime
import time
import platform
from scripts.SearchImage import AgenticReport
from scripts.XAI.explanation.visualize import generate_xai_visualizations
from scripts.HuggingFake import query

# Import LipSync related modules
from scripts.LipSync.model import LIPINC_model
from scripts.LipSync.utils import get_color_structure_frames

# Define Video_Short exception class (this should ideally be imported from LipSync module)
class Video_Short(Exception):
    """Exception raised when video is too short for LipSync analysis."""
    pass

# Function to get descriptive result from LipSync analysis
def get_result_description(real_p):
    if real_p <= 1 and real_p >= 0.99:
        return 'This sample is certainly real.'
    elif real_p < 0.99 and real_p >= 0.75:
        return 'This sample is likely real.'
    elif real_p < 0.75 and real_p >= 0.25:
        return 'This sample is maybe real.'
    elif real_p < 0.25 and real_p >= 0.01:
        return 'This sample is unlikely real.'
    elif real_p < 0.01 and real_p >= 0:
        return 'There is no chance that the sample is real.'
    else:
        return 'Error'

def getc2pa(image):
    try:
        # Create a reader from a file path
        reader = c2pa.Reader.from_file(image)
        return reader.json()
    except Exception as err:
        return {"error": str(err)}


def convert_blob_to_image(blob):
    """Converts a blob to an image and saves it."""
    # Convert blob to bytes-like object
    byte_stream = io.BytesIO(blob)
    # Open image using Pillow
    image = Image.open(byte_stream)
    # Save the image
    image.save("output_image.jpg")


def extract_frames(video_path, output_dir="frames", interval=1):
    """
    Extract frames from a video file at specified intervals
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Base directory to save frames
        interval (int): Interval in seconds between frames
    
    Returns:
        tuple: (Path to the created output directory, video FPS, total frames extracted)
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
    
    # Calculate frame step based on interval
    frame_step = int(fps * interval)
    
    frame_count = 0
    saved_count = 0
    
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
        
        frame_count += 1
    
    video.release()
    # Ensure the video is closed before returning
    del video
    
    # Give the OS a moment to fully release the file handle (especially important on Windows)
    
    return output_dir, fps, saved_count


def safe_delete_file(file_path, max_attempts=5, delay=1.0):
    """
    Safely delete a file with multiple attempts, handling Windows file lock issues
    
    Args:
        file_path (str): Path to the file to delete
        max_attempts (int): Maximum number of deletion attempts
        delay (float): Delay between attempts in seconds
    
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    if not os.path.exists(file_path):
        return True
        
    for attempt in range(max_attempts):
        try:
            os.unlink(file_path)
            return True
        except PermissionError:
            if attempt < max_attempts - 1:
                continue
            else:
                # If we can't delete it after max attempts, just report it but don't fail
                print(f"Warning: Could not delete temporary file {file_path}")
                return False


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/', methods=['GET'])
def handle_root():
    return 'backend api'


@app.route('/extension.crx')
def send_report():
    return send_file('fence_ai.crx')


@app.route('/detect_image', methods=['POST'])
def handle_detect_image():
    if 'file' in request.files:
        file = request.files['file']
        filename = file.filename if file.filename else ''
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
        file.save(temp_file.name)
        manifest = getc2pa(temp_file.name)
        img = cv2.imread(temp_file.name)
        file_extension = file.filename.split('.')[-1].lower()
        
        if file_extension in ['jpg', 'jpeg', 'png', 'webp']:
            # Get model score 
            model_score = query(temp_file.name)
            print(f"Model raw score: {model_score}")

            # Process manifest
            if isinstance(manifest, str):
                try:
                    manifest_data = json.loads(manifest)
                except json.JSONDecodeError:
                    manifest_data = manifest
            else:
                manifest_data = manifest
            manifest_str = json.dumps(manifest) if not isinstance(manifest, str) else manifest

            # Generate visualizations and get report
            segmented = generate_xai_visualizations(temp_file.name, r"/teamspace/studios/this_studio/Guard.ai/backend/scripts/XAI/model/checkpoint/ff_attribution.ckpt")
            report = AgenticReport(temp_file.name, "fake", manifest_str, output=model_score)
            print(f"Report: {report}")
           
            llm_score = report.get('llm_score', 0.5) # Default to 0.5 if not found
            print(f"LLM analysis score: {llm_score}")
            print(llm_score)
            print(model_score)
            # Calculate combined score
            combined_score = (0.4 * llm_score) + (0.6 * model_score[1]['score'])
            print(f"Combined score: {combined_score}")

            # Clean up
            safe_delete_file(temp_file.name)

            # Return response with scores
            print(model_score)
            return {
                # 'model_score': float(model_score[1]['score']),
                # 'llm_score': float(llm_score), 
                # 'combined_score': float(combined_score),
                'deepfake': model_score, #[1]['score'],
                'manifest': manifest_data,
                'report': report,
                'segmented': segmented
            }
        else:
            safe_delete_file(temp_file.name)
            return f'Unsupported file format: {file_extension}'
    else:
        return 'No image data received'

@app.route('/detect_video', methods=['POST'])
def handle_detect_video():
    temp_video = None
    frames_dir = None
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No video file received'}), 400
        
        file = request.files['file']
        filename = file.filename if file.filename else ''
        file_extension = filename.split('.')[-1].lower()
        
        if file_extension not in ['mp4', 'avi', 'mov', 'mkv', 'webm']:
            return jsonify({'error': f'Unsupported file format: {file_extension}'}), 400
        
        # Save video to temp file
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}")
        temp_video_path = temp_video.name
        temp_video.close()
        file.save(temp_video_path)
        
        # --- Lip-Sync Analysis ---
        lip_sync_result = None
        try:
            # Extract lip features
            length_error, _, combined_frames, residue_frames, _, _ = get_color_structure_frames(
                5, temp_video_path  # n_frames=5
            )
            
            if not length_error:
                # Reshape for model input
                combined_frames = np.reshape(combined_frames, (1,) + combined_frames.shape)
                residue_frames = np.reshape(residue_frames, (1,) + residue_frames.shape)
                
                # Predict
                prediction = lip_sync_model.predict([combined_frames, residue_frames])
                real_prob = round(float(prediction[0][1]), 3)
                lip_sync_result = {
                    'real_probability': real_prob,
                    'fake_probability': round(1 - real_prob, 3),
                    'is_fake': real_prob < 0.5  # Adjust threshold as needed
                }
        except Exception as e:
            print(f"Lip-sync analysis failed: {str(e)}")
        
        # --- Existing Frame-by-Frame Analysis ---
        interval = int(request.form.get('interval', 1))
        frames_dir, fps, frame_count = extract_frames(temp_video_path, interval=interval)
        
        results = []
        fake_timestamps = []
        
        # Load the LipSync model
        lip_sync_model = LIPINC_model()
        checkpoint_path = "scripts/LipSync/checkpoints/FakeAv.hdf5"
        lip_sync_model.load_weights(checkpoint_path)
        
        for i in range(1, frame_count + 1):
            frame_path = os.path.join(frames_dir, f"frame_{i}.jpg")
            
            # Get C2PA manifest
            manifest = getc2pa(frame_path)
            
            # Load and analyze the image
            img = cv2.imread(frame_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(img_rgb)
            
            predictions = analyze_image(face_pil)
            fusion = predictions.get('fusion')
            
            # Calculate timestamp for this frame
            timestamp_seconds = (i - 1) * interval
            timestamp = str(timedelta(seconds=timestamp_seconds))
            
            # Determine if frame is fake
            is_fake = False
            if fusion is not None and (fusion > 0 or (-0.000001 <= fusion <= -0.23)):
                is_fake = True
                fake_timestamps.append({
                    'timestamp': timestamp,
                    'frame_number': i,
                    'fusion_score': fusion
                })
            
            # Add result for this frame
            results.append({
                'frame_number': i,
                'timestamp': timestamp,
                'prediction': "Fake" if is_fake else "Real",
                'fusion_score': fusion,
                'is_fake': is_fake
            })
        
        # LipSync processing
        n_frames = 5  # number of local frames
        length_error, face, combined_frames, residue_frames, l_id, g_id = get_color_structure_frames(n_frames, temp_video_path)
        if length_error:
            raise Video_Short()
        
        combined_frames = np.reshape(combined_frames, (1,) + combined_frames.shape)
        residue_frames = np.reshape(residue_frames, (1,) + residue_frames.shape)
        
        lip_sync_result = lip_sync_model.predict([combined_frames, residue_frames])
        lip_sync_result = round(float(lip_sync_result[0][1]), 3)
        
        # Return the analysis results
        response_data = {
            'video_analysis': {
                'total_frames_analyzed': frame_count,
                'frame_interval': interval,
                'fps': fps,
                'duration': str(timedelta(seconds=frame_count * interval)),
                'results': results,
                'fake_frames_detected': len(fake_timestamps),
                'fake_timestamps': fake_timestamps
            },
            'lip_sync_analysis': lip_sync_result if lip_sync_result else "Analysis failed"
        }
        
        if frames_dir and os.path.exists(frames_dir):
            shutil.rmtree(frames_dir, ignore_errors=True)
        
        # Safely delete the temp video file
        if temp_video_path and os.path.exists(temp_video_path):
            safe_delete_file(temp_video_path)
        return jsonify(response_data)
        
    except Exception as e:
        # Clean up on error
        if frames_dir and os.path.exists(frames_dir):
            shutil.rmtree(frames_dir, ignore_errors=True)
            
        if temp_video and hasattr(temp_video, 'name') and os.path.exists(temp_video.name):
            safe_delete_file(temp_video.name)
        
        return jsonify({'error': str(e)}), 500

from scripts.Report import generate_pdf_report, generate_case_number
import os
import time

# Create directory for reports if it doesn't exist
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

@app.route('/generate_report', methods=['POST'])
def handle_generate_report():
    try:
        if request.method == 'OPTIONS':
            # Handle preflight request
            response = app.make_default_options_response()
            return response
            
        # Get input data
        if request.is_json:
            data = request.json
        else:
            data = request.form.to_dict()
            
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Required fields
        investigator_name = data.get('investigator_name', 'Unknown Investigator')
        case_number = data.get('case_number') or generate_case_number()
        
        # Check if we have an image file or path
        image_path = None
        if 'image_path' in data:
            image_path = data.get('image_path')
        elif 'file' in request.files:
            # Save the uploaded file
            file = request.files['file']
            filename = file.filename if file.filename else f"case_{case_number}.jpg"
            temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
            file.save(temp_image.name)
            image_path = temp_image.name
        
        # Get analysis results
        analysis_results = data.get('analysis_results')
        if not analysis_results and 'results' in data:
            analysis_results = data.get('results')
            
        # If analysis_results is a string, parse it to JSON
        if isinstance(analysis_results, str):
            try:
                analysis_results = json.loads(analysis_results)
            except json.JSONDecodeError:
                pass  # Keep it as a string if parsing fails
            
        # Grad-CAM path (optional)
        grad_cam_path = data.get('grad_cam_path')
            
        # Validate required inputs
        if not image_path:
            return jsonify({'error': 'No image provided (either as path or file upload)'}), 400
        if not analysis_results:
            return jsonify({'error': 'No analysis results provided'}), 400
            
        # Check if image exists
        if not os.path.exists(image_path):
            return jsonify({'error': f'Image file does not exist at path: {image_path}'}), 400
        
        # Generate report PDF
        timestamp = int(time.time())
        pdf_filename = f"DeepfakeReport_{case_number}_{timestamp}.pdf"
        pdf_path = os.path.join(REPORTS_DIR, pdf_filename)
        
        # Print debug info
        print(f"Generating report to: {pdf_path}")
        print(f"Input image: {image_path}")
        print(f"Case number: {case_number}")
        
        # Call the report generator
        generate_pdf_report(
            output_path=pdf_path,
            case_number=case_number,
            investigator_name=investigator_name,
            analysis_results=analysis_results,
            image_path=image_path,
            grad_cam_path=grad_cam_path
        )
        
        # Clean up temporary file if we created one
        if 'temp_image' in locals() and os.path.exists(temp_image.name):
            try:
                os.unlink(temp_image.name)
            except Exception as e:
                print(f"Warning: Could not delete temporary file: {e}")
        
        # Return the PDF file
        print(f"Sending file: {pdf_path}")
        return send_file(
            pdf_path, 
            as_attachment=True, 
            download_name=pdf_filename,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
lip_sync_model = LIPINC_model()
lip_sync_model.load_weights(r"/teamspace/studios/this_studio/Guard.ai/backend/scripts/LipSync/checkpoints/FakeAv.hdf5")
