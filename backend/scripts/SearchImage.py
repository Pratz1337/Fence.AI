import os
import json
from typing import List, Union, Dict, Any
from base64 import b64decode

from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.pydantic_v1 import BaseModel, Field

# Replace TavilySearchResults import as needed
from langchain_community.tools.tavily_search import TavilySearchResults

# For image encoding/processing
import base64
from io import BytesIO
from PIL import Image
import pytesseract
import cv2
import numpy as np
import requests

# Import Groq client directly
from groq import Groq

# Set your API keys
os.environ["TAVILY_API_KEY"] = "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"
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual Groq API key

# Import Groq LLM integration for agent reasoning
from langchain_groq import ChatGroq

# Function to encode image for API usage
def encode_image(image_path):
    """Encode an image to a base64 string from a file path or URL"""
    if image_path.startswith(('http://', 'https://')):
        response = requests.get(image_path)
        image_data = response.content
    else:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
    return base64.b64encode(image_data).decode('utf-8')

# Function to decode base64 image to a temporary file
def decode_base64_to_image(base64_string, output_path="temp_image.jpg"):
    """Decode a base64 string to an image file"""
    try:
        image_data = base64.b64decode(base64_string)
        with open(output_path, 'wb') as f:
            f.write(image_data)
        return output_path
    except Exception as e:
        print(f"Error decoding base64 image: {str(e)}")
        return None

# Function to download an image from URL to a local file if needed
def download_image_if_needed(image_url, local_path="temp_image.jpg"):
    """Download an image from URL to a local file if needed."""
    if image_url.startswith(('http://', 'https://')):
        try:
            response = requests.get(image_url)
            with open(local_path, 'wb') as f:
                f.write(response.content)
            return local_path
        except Exception as e:
            print(f"Error downloading image: {str(e)}")
            return None
    return image_url  # Return the original path if it's already local

# Image analysis tool for anomaly detection
class AnomalyDetector:
    def analyze_image(self, image_path: str) -> str:
        """
        Analyze an image for visual anomalies using a basic computer vision approach.
        """
        try:
            local_path = download_image_if_needed(image_path)
            if not local_path:
                return "Error: Could not download or access the image."
            img = cv2.imread(local_path)
            if img is None:
                return "Error: Could not load the image for analysis."
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            height, width = img.shape[:2]
            total_area = height * width
            unusual_shapes = []
            large_areas = []
            for contour in contours:
                area = cv2.contourArea(contour)
                area_percentage = (area / total_area) * 100
                if area_percentage > 2:
                    large_areas.append(area_percentage)
                if area > 500:
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                    if len(approx) > 8:
                        unusual_shapes.append(len(approx))
            color_channels = cv2.mean(img)[:3]
            brightness = sum(color_channels) / 3
            results = [
                f"Image dimensions: {width}x{height} pixels",
                f"Detected {len(contours)} distinct contours/objects",
                f"Found {len(unusual_shapes)} complex shapes with unusual contours: {unusual_shapes if unusual_shapes else 'None'}",
                f"Detected {len(large_areas)} significant objects covering at least 2% of the image each",
                f"Average brightness: {brightness:.1f}/255 ({brightness/255*100:.1f}%)"
            ]
            if unusual_shapes:
                results.append("ANOMALY DETECTED: Unusual shapes found in the image.")
            if brightness < 50:
                results.append("ANOMALY DETECTED: Image appears unusually dark.")
            elif brightness > 200:
                results.append("ANOMALY DETECTED: Image appears unusually bright.")
            return "\n".join(results)
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

# Text extraction tool using OCR
class TextExtractor:
    def extract_text(self, image_path: str) -> str:
        """
        Extract text from an image using OCR.
        """
        try:
            local_path = download_image_if_needed(image_path)
            if not local_path:
                return "Error: Could not download or access the image."
            img = Image.open(local_path)
            img_gray = img.convert('L')
            img_contrast = Image.fromarray(cv2.equalizeHist(np.array(img_gray)))
            text = pytesseract.image_to_string(img)
            if not text.strip():
                text = pytesseract.image_to_string(img_contrast)
            if not text.strip():
                img_array = np.array(img_gray)
                _, img_threshold = cv2.threshold(img_array, 150, 255, cv2.THRESH_BINARY)
                text = pytesseract.image_to_string(Image.fromarray(img_threshold))
            if text.strip():
                return f"Text detected in image:\n{text}"
            else:
                return "No text could be detected in the image after multiple OCR attempts."
        except Exception as e:
            return f"Error extracting text: {str(e)}"

# Groq Vision API integration
class GroqVisionAnalyzer:
    def __init__(self):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = "llama-3.2-11b-vision-preview"
        
    def analyze_image_content(self, image_path: str) -> str:
        """
        Analyze image content using Groq's Vision API.
        """
        try:
            local_path = download_image_if_needed(image_path)
            if not local_path:
                return "Error: Could not download or access the image."
            base64_image = encode_image(local_path)
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this image in detail. Describe what you see, including any objects, people, text, and notable elements. Then, based solely on this analysis, determine if the image is REAL or FAKE. Provide a clear final verdict with your reasoning."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            },
                        ],
                    }
                ],
                model=self.model,
                max_tokens=500
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error analyzing image with Groq Vision API: {str(e)}"
            
    def analyze_explanation_images(self, original_image_path: str, lime_image_base64: str, gradcam_image_base64: str) -> str:
        """
        Analyze original image along with LIME and Grad-CAM visualization images.
        
        Parameters:
        - original_image_path: Path to the original image to analyze
        - lime_image_base64: Base64 string of the LIME explanation image
        - gradcam_image_base64: Base64 string of the Grad-CAM explanation image
        
        Returns:
        A detailed analysis of the image authenticity based on the model explanations
        """
        try:
            # Save the base64 images to temporary files
            lime_path = decode_base64_to_image(lime_image_base64, "temp_lime.jpg")
            gradcam_path = decode_base64_to_image(gradcam_image_base64, "temp_gradcam.jpg")
            
            if not lime_path or not gradcam_path:
                return "Error: Could not decode the explanation images."
                
            # Ensure we have the original image in a local path
            local_original_path = download_image_if_needed(original_image_path, "temp_original.jpg")
            if not local_original_path:
                return "Error: Could not download or access the original image."
                
            # Encode all three images for the API
            base64_original = encode_image(local_original_path)
            base64_lime = encode_image(lime_path)
            base64_gradcam = encode_image(gradcam_path)
            
            # Create a comprehensive prompt for the vision model to analyze all images
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": """I'll show you three related images:
1. The original image to be analyzed
2. A LIME explanation visualization of the image (shows regions that influenced the AI model's decision with bright areas)
3. A Grad-CAM explanation visualization (highlights areas that most activated the AI model, with warmer colors indicating higher activation)

Based on all three images together, analyze:
1. What the original image depicts
2. What areas the LIME image highlights as suspicious
3. What the Grad-CAM visualization reveals about model attention
4. Whether the highlighted regions in LIME and Grad-CAM correspond to visually irregular areas in the original
5. Based on your analysis of ALL THREE images, determine if the original image is REAL or FAKE. Provide a detailed explanation focusing on visual inconsistencies in lighting, textures, facial features, etc.

Provide a comprehensive analysis and a clear final verdict."""},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_original}"}
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_lime}"}
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_gradcam}"}
                            },
                        ],
                    }
                ],
                model=self.model,
                max_tokens=1000
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error analyzing explanation images: {str(e)}"

# Comprehensive image processor that combines all analysis methods
class ComprehensiveImageProcessor:
    def __init__(self):
        self.vision_analyzer = GroqVisionAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.text_extractor = TextExtractor()
        
    def process_image_with_explanations(self, image_path: str, lime_base64: str, gradcam_base64: str, metadata: str) -> str:
        """
        Process an image using all available tools along with LIME and Grad-CAM explanations.
        
        Parameters:
        - image_path: URL or local path to the original image.
        - lime_base64: Base64 string of the LIME explanation image.
        - gradcam_base64: Base64 string of the Grad-CAM explanation image.
        - metadata: Metadata of the image (including C2PA data).
        
        Returns:
        A JSON-formatted string containing the comprehensive analysis results.
        """
        try:
            local_path = download_image_if_needed(image_path)
            if not local_path:
                return json.dumps({"error": "Could not download or access the image."})
            
            print("Processing explanation images...")
            explanation_analysis = self.vision_analyzer.analyze_explanation_images(
                local_path, lime_base64, gradcam_base64
            )
            
            print("Processing visual content...")
            vision_analysis = self.vision_analyzer.analyze_image_content(local_path)
            
            print("Detecting anomalies...")
            anomaly_analysis = self.anomaly_detector.analyze_image(local_path)
            print(anomaly_analysis)
            
            print("Extracting text...")
            text_analysis = self.text_extractor.extract_text(local_path)
            print(text_analysis)
            
            search_context = ""
            if "Text detected in image:" in text_analysis:
                extracted = text_analysis.replace("Text detected in image:\n", "").strip()
                search_response = TavilySearchResults(max_results=3).invoke(extracted)
                search_context = f"{search_response}"
            
            report = {
                "Visual Content Analysis": vision_analysis,
                "Explanation Models Analysis": explanation_analysis,
                "Anomaly Detection": anomaly_analysis,
                "Text Extraction": text_analysis,
                "Image Metadata": metadata,
                "Additional Context": search_context if search_context else "No additional context obtained from web search.",
                "Final Summary and Verdict": ""  # Placeholder; agent should generate this
            }
            
            print(report)
            print(json.dumps(report, indent=2))
            return json.dumps(report, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Error during comprehensive image processing: {str(e)}"})

# Create instances of our tools
anomaly_detector = AnomalyDetector()
text_extractor = TextExtractor()
groq_vision_analyzer = GroqVisionAnalyzer()
comprehensive_processor = ComprehensiveImageProcessor()
search_tool = TavilySearchResults(max_results=3)

# Create tools for the agent
tools = [
    Tool(
        name="ComprehensiveImageExplanationAnalysis",
        func=comprehensive_processor.process_image_with_explanations,
        description="Performs a comprehensive analysis of an image including original image analysis along with LIME and Grad-CAM explanation visualizations. Pass the image path or URL, LIME image base64, Grad-CAM image base64, and metadata."
    ),
    Tool(
        name="ExplanationImagesAnalysis",
        func=groq_vision_analyzer.analyze_explanation_images,
        description="Analyzes original image along with LIME and Grad-CAM visualization images to determine authenticity. Pass the original image path, LIME image base64, and Grad-CAM image base64."
    ),
    Tool(
        name="GroqVisionAnalysis",
        func=groq_vision_analyzer.analyze_image_content,
        description="Analyzes the visual content of an image using Groq's Vision API. Pass the image path or URL."
    ),
    Tool(
        name="AnomalyDetection",
        func=anomaly_detector.analyze_image,
        description="Analyzes an image to detect visual anomalies. Pass the image path or URL."
    ),
    Tool(
        name="TextExtraction",
        func=text_extractor.extract_text,
        description="Extracts text from an image using OCR. Pass the image path or URL."
    ),
    Tool(
        name="WebSearch",
        func=search_tool.invoke,
        description="Searches the web for information. Use when additional context is needed based on extracted text."
    )
]

# Create a text LLM for the agent's reasoning using Groq's text model
agent_llm = ChatGroq(
    model="llama3-70b-8192",  # Using one of Groq's text models
    temperature=0
)

# Create the agent prompt with explicit JSON instructions for final output
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an intelligent multimodal agent that analyzes images comprehensively with a focus on detecting fake or manipulated images.

When given an original image, LIME visualization, Grad-CAM visualization, and metadata, ALWAYS use the ComprehensiveImageExplanationAnalysis tool first.
This tool will:
1. Analyze the original image for visual content using Groq's Vision API.
2. Interpret the LIME and Grad-CAM visualizations to understand what regions influenced the model's decision.
3. Detect anomalies in the original image using computer vision techniques.
4. Extract any text present using OCR.

After obtaining these results, combine them with the provided image metadata.
If additional context is needed based on the extracted text, perform a web search.

Pay special attention to:
- Areas highlighted in LIME and Grad-CAM visualizations and whether they correspond to actual visual inconsistencies
- Inconsistencies in lighting, shadows, textures, and proportions in the original image
- Unusual artifacts around edges of objects, particularly faces
- Unnatural blurring or sharpness patterns

Now, based solely on all the evidence, generate a final report as a JSON object with the following keys exactly:
- "Visual Content Analysis"
- "Explanation Models Analysis"
- "Anomaly Detection"
- "Text Extraction"
- "Image Metadata"
- "Additional Context"
- "Final Summary and Verdict"

For the "Final Summary and Verdict", do not simply repeat the above information. Instead, analyze and synthesize a conclusion that clearly states whether the image is REAL or FAKE, along with a concise explanation of your reasoning focused on the visual inconsistencies and the explanation model outputs. Return only the JSON object with these keys and no extra text.
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent using the tools and prompt
agent = create_openai_functions_agent(agent_llm, tools, prompt)

# Create the agent executor with increased iterations if needed
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    return_intermediate_steps=True
)

# Define a function to be imported and called from another file
def AgenticReport(image_path: str, lime_base64: str, gradcam_base64: str, metadata: str = "",output: str="") -> Dict[str, Any]:
    """
    Process an image through the multimodal agent with LIME and Grad-CAM explanations.

    Parameters:
    - image_path: URL or local path to the original image.
    - lime_base64: Base64 string of the LIME explanation image.
    - gradcam_base64: Base64 string of the Grad-CAM explanation image.
    - metadata: Metadata of the image (including C2PA data).

    Returns:
    A dictionary with the final comprehensive report.
    """
    try:
        local_path = download_image_if_needed(image_path)
        if not local_path:
            return {"error": "Could not download or access the image."}
        
        input_text = (
            f"Analyze this image thoroughly: {local_path}\n"
            f"LIME explanation image provided as base64\n"
            f"Grad-CAM explanation image provided as base64\n"
            f"Output: {output}\n"
            f"Metadata: {metadata}"
        )
        
        # Add fallback if agent initialization fails
        try:
            result = agent_executor.invoke({
                "input": input_text,
                "chat_history": []
            })
            # Rest of the agent processing...
        except Exception as agent_error:
            print(f"Agent executor failed: {str(agent_error)}")
            # Fallback to direct processor
            direct_result = json.loads(comprehensive_processor.process_image_with_explanations(
                image_path, lime_base64, gradcam_base64, metadata
            ))
            direct_result["Final Summary and Verdict"] = agent_llm.invoke(
                f"Based on this evidence, state if the image is REAL or FAKE with reasoning, also consider the Output array of the Model based detection carried out which is {output}:\n" +
                json.dumps(direct_result, indent=2)
            ).content
            return direct_result
            
        print("Full Result Dictionary:")
        print(result)
        final_output = result.get("output", "").strip()
        if not final_output:
            print("Agent produced no output. Falling back to direct processor.")
            direct_result = comprehensive_processor.process_image_with_explanations(
                image_path, lime_base64, gradcam_base64, metadata
            )
            final_output = direct_result
        try:
            final_report = json.loads(final_output)
            print(final_report)
        except Exception as parse_e:
            return {"error": f"Final output is not valid JSON: {parse_e}", "raw_output": final_output}
        
        # If the "Final Summary and Verdict" key is empty, generate a follow-up using the evidence.
        if not final_report.get("Final Summary and Verdict", "").strip():
            evidence = "\n".join([
                "Visual Content Analysis: " + final_report.get("Visual Content Analysis", ""),
                "Explanation Models Analysis: " + final_report.get("Explanation Models Analysis", ""),
                "Anomaly Detection: " + final_report.get("Anomaly Detection", ""),
                "Text Extraction: " + final_report.get("Text Extraction", ""),
                "Image Metadata: " + final_report.get("Image Metadata", ""),
                "Additional Context: " + final_report.get("Additional Context", "")
            ])
            follow_up_prompt = (
                "Based on the above evidence, particularly the LIME and Grad-CAM explanation visualizations, "
                "please generate a clear final summary and verdict that states whether the image is REAL or FAKE "
                "and provides a concise explanation of your reasoning. Focus on identifying visual inconsistencies "
                "in lighting, textures, facial features, etc. that correspond to the highlighted regions in the "
                "explanation visualizations."
            )
            full_follow_up = evidence + "\n" + follow_up_prompt
            generated_summary = agent_llm.invoke(full_follow_up)
            final_report["Final Summary and Verdict"] = generated_summary.content
        
        return final_report
    except Exception as e:
        print(f"AgenticReport error: {str(e)}")
        return {
            "Visual Content Analysis": "Analysis failed due to an error",
            "Explanation Models Analysis": "Failed to analyze LIME and Grad-CAM visualizations",
            "Anomaly Detection": "Analysis failed due to an error",
            "Text Extraction": "Analysis failed due to an error",
            "Image Metadata": "Failed to process metadata",
            "Additional Context": "No additional context available due to error",
            "Final Summary and Verdict": "Analysis failed due to an error. Unable to determine if the image is real or fake."
        }