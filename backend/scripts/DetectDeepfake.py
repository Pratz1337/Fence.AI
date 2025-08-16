from PIL import Image
from scripts.clip.Detect import analyze_image

import numpy as np

def predict(face):
    # Convert numpy array to PIL Image
    if isinstance(face, np.ndarray):
        # Ensure the array has the right data type for an image
        if face.dtype != np.uint8:
            # Convert to uint8 if needed (assuming values in [0,1])
            if face.max() <= 1.0:
                face = (face * 255).astype(np.uint8)
        face_pil = Image.fromarray(face)
        return analyze_image(face_pil)
    else:
        # If already a PIL Image, pass directly
        return analyze_image(face)


# def predict_face(pil_images):
#     preds = []
#     for pil_image in pil_images:
#         pred = predict(pil_image)
#         if pred != -1:
#             preds.append(pred)
#     return preds


# def predict_face_from_video(faces_array):
#     predictions = []
#     for face in faces_array:
#         img = Image.fromarray(face)
#         preds = predict(img)
#         if preds != -1:
#             predictions.append(preds)
#     return predictions
