from transformers import pipeline
from PIL import Image

def query(image_path):
    """
    Detect if an image is a deepfake using a ViT model.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Dictionary containing classification results with labels and scores
    """
    # Initialize the pipeline
    pipe = pipeline("image-classification", model="Wvolf/ViT_Deepfake_Detection")
    
    # Load the image
    try:
        image = Image.open(image_path)
    except Exception as e:
        return {"error": f"Failed to open image: {str(e)}"}
    
    # Run the classification
    try:
        results = pipe(image)
        return results
    except Exception as e:
        return {"error": f"Classification failed: {str(e)}"}