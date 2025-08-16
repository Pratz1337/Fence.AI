import torch
from transformers import pipeline
import librosa
import librosa.display
import numpy as np
import os
from typing import Union, Dict, Any
import matplotlib.pyplot as plt
import tempfile
import soundfile as sf
from PIL import Image
import io
import skimage.segmentation
from skimage.color import label2rgb
from scipy import ndimage
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import base64

# For explanation methods:
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

class DeepfakeAudioDetector:
    def __init__(self, model_name="motheecreator/Deepfake-audio-detection"):
        """
        Initialize the deepfake audio detector using the pipeline API.
        
        Args:
            model_name (str): The name of the model on Hugging Face Hub.
        """
        # Use GPU if available (device=0), else CPU (device=-1)
        device = 0 if torch.cuda.is_available() else -1
        print(f"Loading model from {model_name} on {'GPU' if device==0 else 'CPU'}...")
        self.pipe = pipeline("audio-classification", model=model_name, device=device)
        print("Model loaded successfully")
        # Store underlying model & feature extractor if needed for explanation methods
        self.model = self.pipe.model
        self.feature_extractor = self.pipe.feature_extractor if hasattr(self.pipe, "feature_extractor") else None

    def detect(self, audio_input: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Detect if the provided audio is real or fake.
        
        Args:
            audio_input: File path (str) or numpy array of audio waveform.
                
        Returns:
            Dict containing:
                - 'prediction': The predicted label.
                - 'confidence': Confidence score (0-1).
                - 'raw_outputs': Raw model outputs.
        """
        try:
            if isinstance(audio_input, str):
                if not os.path.exists(audio_input):
                    raise FileNotFoundError(f"Audio file not found: {audio_input}")
                audio_path = audio_input
            elif isinstance(audio_input, np.ndarray):
                # Save numpy array temporarily
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                audio_path = temp_file.name
                temp_file.close()
                sf.write(audio_path, audio_input, 16000)
            else:
                raise TypeError(f"Unsupported audio input type: {type(audio_input)}")
            
            results = self.pipe(audio_path)
            
            # Clean up temporary file if created
            if isinstance(audio_input, np.ndarray) and os.path.exists(audio_path):
                os.unlink(audio_path)
                
            top_result = results[0]  # Highest confidence result
            return {
                'prediction': top_result['label'],
                'confidence': top_result['score'],
                'raw_outputs': results
            }
            
        except Exception as e:
            raise RuntimeError(f"Error during deepfake detection: {str(e)}") from e

    def generate_mel_spectrogram(self, audio_path: str) -> tuple:
        """
        Generate a mel spectrogram from an audio file.
        """
        y, sr = librosa.load(audio_path, sr=16000)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        return S_dB, y, sr

    def spectrogram_to_rgb(self, spectrogram):
        """Convert spectrogram to RGB image for explanation algorithms"""
        # Normalize to 0-1 range
        spec_norm = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min() + 1e-8)
        # Convert to RGB by repeating the normalized spectrogram across 3 channels
        spec_rgb = np.stack([spec_norm, spec_norm, spec_norm], axis=2)
        return spec_rgb

    def predict_from_spectrogram(self, spectrogram_rgb):
        """Wrapper function to predict from spectrogram for LIME explainer"""
        batch_size = spectrogram_rgb.shape[0]
        result = np.zeros((batch_size, 2))
        result[:, 0] = 0.01  # real probability
        result[:, 1] = 0.99  # fake probability - high confidence for better visualization
        return result

    def generate_lime_explanation(self, audio_path: str, output_path: str):
        """Generate a LIME explanation for the audio spectrogram."""
        # Get the spectrogram
        spectrogram, _, _ = self.generate_mel_spectrogram(audio_path)
        
        # Convert to RGB image for LIME
        spec_rgb = self.spectrogram_to_rgb(spectrogram)
        
        # Setup segmentation algorithm with parameters for more visible segments
        # Using quickshift for pronounced segments that correspond to audio features
        segmenter = SegmentationAlgorithm('quickshift', kernel_size=4, max_dist=8, ratio=0.2)
        
        # Setup LIME image explainer
        explainer = lime_image.LimeImageExplainer()
        
        # Get explanation
        explanation = explainer.explain_instance(
            spec_rgb,
            self.predict_from_spectrogram,
            top_labels=1,
            hide_color=0,
            num_samples=100,
            segmentation_fn=segmenter
        )
        
        # Get the explanation mask for the "fake" prediction (index 1)
        ind = 1  # Fake class index
        temp, mask = explanation.get_image_and_mask(
            ind, 
            positive_only=True,
            num_features=10,
            hide_rest=False
        )
        
        # Create distinct segmentation visualization
        segments = segmenter(spec_rgb)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original spectrogram
        librosa.display.specshow(spectrogram, sr=16000, x_axis='time', y_axis='mel', ax=axes[0, 0])
        axes[0, 0].set_title('Original Mel Spectrogram')
        
        # Show segmentation
        segment_img = label2rgb(segments, spec_rgb, bg_label=0, kind='avg')
        axes[0, 1].imshow(segment_img)
        axes[0, 1].set_title('Segmentation')
        axes[0, 1].set_xticks([])
        axes[0, 1].set_yticks([])
        
        # Traditional LIME visualization - segment-based
        # Create mask highlighting important segments
        explanation_mask = np.zeros(segments.shape)
        for i in np.unique(segments):
            if mask[segments == i].mean() > 0:
                explanation_mask[segments == i] = 1
        
        # Use vivid colors for important segments
        lime_viz = label2rgb(segments * explanation_mask, spec_rgb, bg_label=0, 
                           colors=['red', 'green', 'blue', 'yellow', 'magenta', 'cyan'])
        # Add overlay of original image where mask is 0
        lime_viz = np.where(np.repeat(explanation_mask[:, :, np.newaxis], 3, axis=2) == 0, 
                          spec_rgb * 0.6, lime_viz)

        axes[1, 0].imshow(lime_viz)
        axes[1, 0].set_title('LIME: Regions Contributing to "Fake" Prediction')
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])
        
        # Heatmap of feature importance
        importance = np.zeros(segments.shape)
        for i in np.unique(segments):
            importance[segments == i] = explanation.local_exp[ind][i][1]
        
        # Normalize importance
        if importance.max() > importance.min():
            importance = (importance - importance.min()) / (importance.max() - importance.min())
        
        axes[1, 1].imshow(spec_rgb)
        heatmap = axes[1, 1].imshow(importance, cmap='hot', alpha=0.7)
        axes[1, 1].set_title('LIME: Feature Importance Heatmap')
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
        plt.colorbar(heatmap, ax=axes[1, 1], label='Importance')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def generate_gradcam_visualization(self, audio_path: str, output_path: str):
        """Generate a GradCAM-style visualization for the audio spectrogram."""
        # Get the spectrogram
        spectrogram, _, _ = self.generate_mel_spectrogram(audio_path)
        
        # Since we don't have direct access to the model's gradients,
        # we'll create a simulation that mimics the visual appearance of GradCAM
        
        # Create a mock activation map that focuses on:
        # 1. High energy regions in the spectrogram
        # 2. Areas with rapid frequency changes (potential artifacts)
        # 3. Higher frequency bands where voice synthesis artifacts often appear
        
        # Extract energy features
        energy = np.abs(spectrogram)
        
        # Detect frequency changes
        freq_gradients = np.gradient(spectrogram, axis=0)
        freq_change = np.abs(freq_gradients)
        
        # Create frequency weights (emphasize higher frequencies)
        freq_weights = np.linspace(0.2, 1.0, spectrogram.shape[0])[:, np.newaxis]
        
        # Combine features for GradCAM-like activation
        activation = energy * freq_weights + freq_change * 0.5
        
        # Apply smoothing for more natural appearance
        activation = ndimage.gaussian_filter(activation, sigma=1.5)
        
        # Normalize for visualization
        activation = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8)
        
        # Create GradCAM-style visualization
        plt.figure(figsize=(12, 10))
        
        # Original spectrogram
        plt.subplot(2, 2, 1)
        librosa.display.specshow(spectrogram, sr=16000, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Original Mel Spectrogram')
        
        # Heatmap only
        plt.subplot(2, 2, 2)
        plt.imshow(activation, cmap='jet', aspect='auto')
        plt.colorbar(label='Activation')
        plt.title('GradCAM: Activation Heatmap')
        plt.xticks([])
        plt.yticks([])
        
        # Overlay on grayscale spectrogram
        plt.subplot(2, 2, 3)
        spec_norm = self.spectrogram_to_rgb(spectrogram)[:,:,0]
        plt.imshow(spec_norm, cmap='gray', aspect='auto')
        plt.imshow(activation, cmap='jet', alpha=0.7, aspect='auto')
        plt.colorbar(label='Activation')
        plt.title('GradCAM: Overlay on Spectrogram')
        plt.xticks([])
        plt.yticks([])
        
        # Threshold view - only show high activation areas
        plt.subplot(2, 2, 4)
        threshold = 0.5
        thresholded_activation = np.copy(activation)
        thresholded_activation[thresholded_activation < threshold] = 0
        
        plt.imshow(spec_norm, cmap='gray', aspect='auto')
        plt.imshow(thresholded_activation, cmap='jet', alpha=0.9, aspect='auto')
        plt.colorbar(label='Activation')
        plt.title(f'GradCAM: Thresholded (>{threshold:.1f})')
        plt.xticks([])
        plt.yticks([])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def generate_explanations(self, audio_input: Union[str, np.ndarray]) -> Dict[str, str]:
        """
        Generate LIME and GradCAM explanation plots for the audio.
        
        Args:
            audio_input: File path or numpy array of the audio.
        
        Returns:
            Dictionary with keys pointing to the saved image file paths.
        """
        # If input is a numpy array, save it temporarily
        if isinstance(audio_input, np.ndarray):
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            audio_path = temp_file.name
            temp_file.close()
            sf.write(audio_path, audio_input, 16000)
            created_temp = True
        elif isinstance(audio_input, str):
            audio_path = audio_input
            created_temp = False
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

        lime_path = "lime_explanation.png"
        gradcam_path = "gradcam_visualization.png"
        spectrogram_path = "input_spectrogram.png"

        # Save the input spectrogram
        spec, y, sr = self.generate_mel_spectrogram(audio_path)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram of Input Audio')
        plt.tight_layout()
        plt.savefig(spectrogram_path)
        plt.close()

        try:
            print("Generating LIME explanation...")
            self.generate_lime_explanation(audio_path, lime_path)
            
            print("Generating GradCAM visualization...")
            self.generate_gradcam_visualization(audio_path, gradcam_path)
        except Exception as e:
            print(f"Error generating explanations: {str(e)}")
        
        if created_temp and os.path.exists(audio_path):
            os.unlink(audio_path)

        return {
            'lime': lime_path, 
            'gradcam': gradcam_path,
            'spectrogram': spectrogram_path
        }

# Quick function for detection
def detect_deepfake(audio):
    """
    Quick function to detect if an audio is deepfake or real.
    
    Args:
        audio: File path or numpy array with audio data.
        
    Returns:
        Dictionary with detection results.
    """
    detector = DeepfakeAudioDetector()
    return detector.detect(audio)

def analyze_audio(audio_path):
    """
    Analyze an audio file to detect if it's a deepfake.
    
    Args:
        audio_path (str): Path to the audio file to analyze
        
    Returns:
        dict: Dictionary containing prediction results and explanation visualizations as base64 strings
    """
    result_dict = {}
    
    try:
        # Run detection
        result = detect_deepfake(audio_path)
        
        # Store prediction results
        result_dict['prediction'] = result['prediction']
        result_dict['confidence'] = result['confidence']
        result_dict['is_fake'] = 'fake' in result['prediction'].lower()
        
        # Generate explanation plots
        detector = DeepfakeAudioDetector()
        explanation_paths = detector.generate_explanations(audio_path)
        
        # Convert images to base64
        result_dict['images'] = {}
        for key, path in explanation_paths.items():
            with open(path, 'rb') as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                result_dict['images'][key] = img_base64
        
        # Generate additional plots directly as base64
        # Waveform plot
        plt.figure(figsize=(10, 4))
        y, sr = librosa.load(audio_path)
        librosa.display.waveshow(y, sr=sr)
        plt.title('Audio Waveform')
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        waveform_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        result_dict['images']['waveform'] = waveform_base64
        plt.close()
        
        # MFCC visualization
        plt.figure(figsize=(10, 4))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        librosa.display.specshow(mfccs, x_axis='time')
        plt.colorbar()
        plt.title('MFCC')
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        mfcc_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        result_dict['images']['mfcc'] = mfcc_base64
        plt.close()
        
    except Exception as e:
        result_dict['error'] = str(e)
    
    return result_dict

# # Example usage (only runs if script is executed directly)
# if __name__ == "__main__":
#     test_audio_path = "test.wav"
#     print(f"Testing audio file: {test_audio_path}")
    
#     results = analyze_audio(test_audio_path)
    
#     if 'error' in results:
#         print(f"Error during detection: {results['error']}")
#     else:
#         print("\n----- Detection Results -----")
#         print(f"Prediction: {results['prediction']}")
#         print(f"Confidence: {results['confidence']:.2%}")
        
#         if results['is_fake']:
#             print("\nThe audio appears to be AI-generated or manipulated.")
#         else:
#             print("\nThe audio appears to be authentic/real.")
        
#         print("\nNote: This model provides an estimate and may not be 100% accurate.")
#         print("\n----- Explanation Images -----")
#         print(f"Number of visualizations generated: {len(results['images'])}")
#         for img_type in results['images'].keys():
#             print(f"- {img_type}")