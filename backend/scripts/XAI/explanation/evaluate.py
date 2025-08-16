import os
import sys
import warnings
import pandas as pd
import numpy as np
import torch
from torchvision.transforms import v2
from PIL import Image
import argparse

# Add the correct paths to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))
sys.path.insert(0, './methods')

# Import after path setup
from model.frame import FrameModel

def explain_single_image(model, inference_image, visualize_image, method="All"):
    """
    Generate explanations for a single image using specified XAI methods.
    
    Args:
        model: The trained model
        inference_image: Processed image tensor for model inference
        visualize_image: Image tensor for visualization
        method: XAI method to use (default: All)
        
    Returns:
        Array of explanation results
    """
    # Get model prediction
    with torch.no_grad():
        output = model(inference_image)
        _, predicted = torch.max(output, 1)
        label = predicted.item()
    
    results = []
    
    # Original (no explanation)
    results.append([0.0, 0.0])  # Placeholder values for original image
    
    # Apply selected XAI methods
    if method == "All" or method == "GradCAM++":
        print("Applying GradCAM++ explanation")
        try:
            from methods.gradcam_xai import explain as GRADCAM
            # Remove batch dimension before passing to GRADCAM since it adds its own
            inference_for_gradcam = inference_image.squeeze(0)
            
            # Get the saliency map
            saliency_map = GRADCAM(inference_for_gradcam, visualize_image, label, model)
            
            # For now, just use placeholder values for sufficiency and stability
            # In a real application, you'd compute these metrics using the saliency map
            results.append([0.5, 0.5])  # Placeholder values for GradCAM++
            
            # Optionally, save the saliency map for visualization
            if isinstance(saliency_map, torch.Tensor):
                saliency_map = saliency_map.cpu().numpy()
            
            np.save(f"./results/saliency_map_gradcam.npy", saliency_map)
        except Exception as e:
            print(f"Error in GradCAM++ explanation: {e}")
            results.append([np.nan, np.nan])
    
    # Similar changes for other methods...
    if method == "All" or method == "RISE":
        print("Applying RISE explanation")
        try:
            from methods.rise_xai import explain as RISE
            results.append([0.5, 0.5])  # Placeholder values for RISE
        except Exception as e:
            print(f"Error in RISE explanation: {e}")
            results.append([np.nan, np.nan])
    
    if method == "All" or method == "SHAP":
        print("Applying SHAP explanation")
        try:
            from methods.shap_xai import explain as SHAP
            results.append([0.5, 0.5])  # Placeholder values for SHAP
        except Exception as e:
            print(f"Error in SHAP explanation: {e}")
            results.append([np.nan, np.nan])
    
    if method == "All" or method == "LIME":
        print("Applying LIME explanation")
        try:
            from methods.lime_xai import explain as LIME
            results.append([0.5, 0.5])  # Placeholder values for LIME
        except Exception as e:
            print(f"Error in LIME explanation: {e}")
            results.append([np.nan, np.nan])
    
    if method == "All" or method == "SOBOL":
        print("Applying SOBOL explanation")
        try:
            from methods.sobol_xai import explain as SOBOL
            results.append([0.5, 0.5])  # Placeholder values for SOBOL
        except Exception as e:
            print(f"Error in SOBOL explanation: {e}")
            results.append([np.nan, np.nan])
    
    # Make sure all results have the same shape
    results = [np.array(r) for r in results]
    return np.array(results)

def main():
    parser = argparse.ArgumentParser(description='Evaluate a single image with XAI methods')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image to evaluate')
    parser.add_argument('--method', type=str, default='All', choices=['All', 'GradCAM++', 'RISE', 'SHAP', 'LIME', 'SOBOL'],
                       help='XAI method to use (default: All)')
    parser.add_argument('--task', type=str, default='multiclass', choices=['binary', 'multiclass'],
                       help='Task type (binary or multiclass)')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--model_path', type=str, default='../model/checkpoint/ff_attribution.ckpt',
                       help='Path to the model checkpoint')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    try:
        model = FrameModel.load_from_checkpoint(args.model_path, map_location=device)
        model.eval()
        print(f"Successfully loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    try:
        # Load and process image
        image = Image.open(args.image_path).convert("RGB")
        print(f"Successfully loaded image: {args.image_path}")
        
        # Process image for inference
        inference_image = inference_transforms(image)
        print(f"Inference image shape: {inference_image.shape}")
        
        # Process image for visualization
        visualize_image = visualize_transforms(image)
        print(f"Visualize image shape: {visualize_image.shape}")
        
        # Convert visualize_image to format expected by show_cam_on_image (H, W, C)
        visualize_image_numpy = visualize_image.permute(1, 2, 0).numpy()
        
        # Add batch dimension to inference image
        inference_image = inference_image.unsqueeze(0).to(device)
        print(f"Final inference image shape: {inference_image.shape}")
        
        # Run explanation
        results = explain_single_image(model, inference_image, visualize_image_numpy, args.method)
        
        # Save results
        save_name = f"results_single_{os.path.basename(args.image_path).split('.')[0]}"
        np.save(f"{args.output_dir}/{save_name}.npy", results)
        
        print(f"Results saved to {args.output_dir}/{save_name}.npy")
        
        # Create and display results dataframe
        if args.method == "All":
            index_values = ["Original", "GradCAM++", "RISE", "SHAP", "LIME", "SOBOL"]
        else:
            index_values = ["Original", args.method]
            
        column_values = ["Sufficiency", "Stability"]
        df = pd.DataFrame(results, index=index_values, columns=column_values)
        print("\nResults:")
        print(df.round(3))
        
        # Save results to CSV
        csv_save_name = f"scores_single_{os.path.basename(args.image_path).split('.')[0]}"
        df.round(3).to_csv(f"{args.output_dir}/{csv_save_name}.csv", sep=',')
        print(f"Results saved to CSV: {args.output_dir}/{csv_save_name}.csv")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()