import os
import sys
import torch
from torchvision.transforms import v2
from PIL import Image
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt
import cv2
import io
import traceback
import base64

current_dir = os.path.dirname(os.path.abspath(__file__))
methods_path = os.path.join(current_dir, 'methods')
if methods_path not in sys.path:
    sys.path.append(methods_path)

# Add the correct paths to sys.path
def setup_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    sys.path.insert(0, os.path.join(parent_dir, 'src'))
    sys.path.insert(0, './methods')

def encode_image_to_base64(image_array):
    """Convert a NumPy array to a base64-encoded image string"""
    # Convert numpy array to PIL Image
    pil_img = Image.fromarray(image_array)
    
    # Save the image to a BytesIO object
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    
    # Encode the BytesIO object to base64
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Return with the proper data URI prefix
    return f"data:image/png;base64,{img_str}"

# Function to generate saliency map visualization
def generate_saliency_visualization(original_image, saliency_map):
    """
    Generate a visualization of the saliency map overlaid on the original image.
    
    Args:
        original_image: The original image as a numpy array (H, W, C)
        saliency_map: The saliency map as a numpy array (H, W)
        
    Returns:
        dict: Dictionary containing 'original', 'saliency', and 'overlay' images as base64 strings
    """
    result = {}
    
    # Ensure saliency map is 2D
    if len(saliency_map.shape) > 2:
        if len(saliency_map.shape) == 3 and saliency_map.shape[2] == 1:
            saliency_map = saliency_map[:, :, 0]
        else:
            # Take mean across channels if multi-channel
            saliency_map = np.mean(saliency_map, axis=2) if saliency_map.shape[2] > 1 else saliency_map[:, :, 0]
    
    # Resize saliency map to match original image dimensions
    h, w = original_image.shape[:2]
    saliency_map = cv2.resize(saliency_map, (w, h))
    
    # Normalize saliency map to [0, 1]
    if saliency_map.max() > saliency_map.min():
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * saliency_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Convert original image to uint8 if it's float
    if original_image.dtype == np.float32 or original_image.dtype == np.float64:
        original_image = (original_image * 255).astype(np.uint8)
    
    # Create overlay
    alpha = 0.4  # Transparency factor
    try:
        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
    except Exception:
        # Fallback if overlay fails
        overlay = original_image.copy()
        overlay = (overlay * 0.7).astype(np.uint8)
        mask = np.stack([saliency_map] * 3, axis=2)
        mask = (mask * 255 * 0.3).astype(np.uint8)
        overlay = overlay + mask
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # Store results as base64
    result['original'] = encode_image_to_base64(original_image)
    result['saliency'] = encode_image_to_base64((saliency_map * 255).astype(np.uint8))
    result['overlay'] = encode_image_to_base64(overlay)
    
    return result

# Main function to generate XAI visualizations
def generate_xai_visualizations(
    image_path, 
    model_path='../model/checkpoint/ff_attribution.ckpt',
    methods=None,
    output_dir='./results',
    save_images=False
):
    """
    Generate XAI visualizations for a given image using multiple methods.
    
    Args:
        image_path (str): Path to the image to visualize
        model_path (str): Path to the model checkpoint
        methods (list): List of XAI methods to use (default: ['GradCAM++', 'LIME'])
        output_dir (str): Directory to save results if save_images is True
        save_images (bool): Whether to save images to disk
        
    Returns:
        dict: Dictionary containing visualization results for each method
    """
    # Setup paths
    setup_paths()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    methods_dir = os.path.join(current_dir, 'methods')

    # Print for debugging
    print(f"Current directory: {current_dir}")
    print(f"Methods directory: {methods_dir}")
    print(f"Methods directory exists: {os.path.exists(methods_dir)}")
    print(f"Methods in sys.path: {methods_dir in sys.path}")
    
    # Add methods directory to path if not already there
    if methods_dir not in sys.path:
        sys.path.insert(0, methods_dir)
    
    # Import model after path setup
    from model.frame import FrameModel
    
    # Define default methods if not provided
    if methods is None:
        methods = ['GradCAM++', 'LIME']
    
    # Create output directory if saving images
    if save_images:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define transforms for inference and visualization
    rs_size = 224
    interpolation = 3
    
    inference_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize(rs_size, interpolation=interpolation, antialias=False),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    visualize_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize(rs_size, interpolation=interpolation, antialias=False),
        v2.ToDtype(torch.float32, scale=True),
    ])
    
    # Load model
    model = FrameModel.load_from_checkpoint(model_path, map_location=device)
    model.eval()
    
    # Load and process image
    original_image = Image.open(image_path).convert("RGB")
    
    # Process image for inference
    inference_image = inference_transforms(original_image)
    
    # Process image for visualization
    visualize_image = visualize_transforms(original_image)
    
    # Convert visualize_image to format expected by visualization methods (H, W, C)
    visualize_image_numpy = visualize_image.permute(1, 2, 0).numpy()
    
    # Compute the inference scores
    inference_image_device = inference_image.to(device)
    
    with torch.no_grad():
        output = model(inference_image_device.unsqueeze(0))
    
    output = output.cpu().reshape(-1, ).numpy()
    
    # Get predicted label
    explanation_label_index = np.argmax(output)
    
    # Dictionary to store results
    results = {
        'prediction': {
            'scores': output.tolist(),
            'label': int(explanation_label_index)
        }
    }
    
    # Generate visualizations for each method
    for method in methods:
        try:
            saliency = None
            
            if method == "GradCAM++":
                from .methods.gradcam_xai import explain as GradCAM
                saliency = GradCAM(inference_image, visualize_image_numpy, explanation_label_index, model)
                
            elif method == "LIME":
                from .methods.lime_xai import explain as LIME
                saliency = LIME(visualize_image_numpy, inference_transforms, explanation_label_index, model)
            
            elif method == "RISE":
                from .methods.rise_xai import explain as RISE
                saliency = RISE(inference_image, visualize_image_numpy, explanation_label_index, model)
                
            elif method == "SHAP":
                from .methods.shap_xai import explain as SHAP
                saliency = SHAP(inference_image, visualize_image_numpy, explanation_label_index, model)
                
            elif method == "SOBOL":
                from .methods.sobol_xai import explain as SOBOL
                saliency = SOBOL(inference_image, visualize_image_numpy, explanation_label_index, model)
            
            else:
                continue
            
            # Convert tensor to numpy if needed
            if isinstance(saliency, torch.Tensor):
                saliency = saliency.cpu().numpy()
            
            # Save saliency map if requested
            if save_images:
                base_filename = os.path.join(output_dir, f"visualization_{method.lower()}")
                np.save(f"{base_filename}_saliency.npy", saliency)
            
            # Generate visualization
            original_np = np.array(original_image.resize((224, 224)))
            visualization = generate_saliency_visualization(original_np, saliency)
            
            # Save visualization if requested
            if save_images:
                visualization_path = f"{base_filename}_visualization.png"
                
                plt.figure(figsize=(18, 6))
                
                plt.subplot(1, 3, 1)
                plt.title("Original Image")
                plt.imshow(visualization['original'])
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.title("Saliency Map")
                plt.imshow(visualization['saliency'], cmap='jet')
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.title("Overlay")
                plt.imshow(visualization['overlay'])
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(visualization_path)
                plt.close()
            
            # Store results
            results[method] = visualization
            
        except Exception as e:
            results[method] = {
                'error': str(e),
                'traceback': traceback.format_exc() if 'traceback' in sys.modules else None
            }
    
    return results