import os  # Added missing import
import io
import json
import uuid
import datetime
import tempfile
import re
from typing import List, Dict, Any, Optional
import traceback
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.platypus import PageBreak, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# LangChain imports with Groq integration
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Import Groq directly
from groq import Groq

# Set Groq API key
os.environ["GROQ_API_KEY"] = "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"

# Langchain Groq integration
from langchain_groq import ChatGroq

def generate_case_number():
    """Generate a unique case number with date prefix"""
    date_prefix = datetime.datetime.now().strftime("%Y%m%d")
    unique_id = str(uuid.uuid4().hex)[:6].upper()
    return f"{date_prefix}-{unique_id}"

def format_report_text(text):
    """Format text with basic markdown parsing for better PDF rendering"""
    if not text:
        return ""
        
    # Replace markdown bold with ReportLab bold tags
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Replace markdown italic with ReportLab italic tags
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    # Handle numbered lists by adding proper spacing
    text = re.sub(r'(\d+)\.\s+', r'<bullet>\1.</bullet> ', text)
    
    return text

def create_header_footer(canvas, doc, case_number, investigator_name):
    """Create header and footer for each page of the report"""
    canvas.saveState()
    
    # Header
    canvas.setFont('Helvetica-Bold', 10)
    canvas.drawString(inch, 10.5 * inch, f"CASE #{case_number}")
    canvas.setFont('Helvetica', 9)
    canvas.drawString(inch, 10.3 * inch, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    canvas.drawString(inch, 10.1 * inch, f"Investigator: {investigator_name}")
    
    # Add a line under the header
    canvas.line(inch, 10*inch, 7.5*inch, 10*inch)
    
    # Footer
    canvas.setFont('Helvetica', 8)
    canvas.line(inch, 0.75*inch, 7.5*inch, 0.75*inch)
    canvas.drawString(inch, 0.5 * inch, "CONFIDENTIAL - VIDEO FORENSIC INVESTIGATION REPORT")
    canvas.drawRightString(7.5 * inch, 0.5 * inch, f"Page {doc.page}")
    
    canvas.restoreState()

def create_frame_analysis_chart(frame_scores, output_path):
    """Create a chart showing deepfake scores across video frames"""
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    
    # Extract frame numbers and scores
    frames = list(range(len(frame_scores)))
    scores = frame_scores
    
    # Create the line plot
    ax.plot(frames, scores, 'b-', linewidth=2)
    ax.fill_between(frames, scores, alpha=0.3, color='blue')
    
    # Add a horizontal line at threshold 0.5
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Deepfake Probability')
    ax.set_title('Deepfake Detection Scores Across Video Frames')
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def analyze_with_llm(video_analysis_results):
    """
    Use LangChain with Groq to analyze video results and generate a comprehensive report
    
    Args:
        video_analysis_results (dict): The results from video deepfake analysis
        
    Returns:
        dict: Structured report data from the LLM
    """
    # Define the output schema for structured output
    response_schemas = [
        ResponseSchema(name="executive_summary", 
                      description="A 2-3 paragraph executive summary of the video analysis findings"),
        ResponseSchema(name="deepfake_assessment", 
                      description="Assessment of whether the video is likely fake or real, with confidence level"),
        ResponseSchema(name="frame_analysis", 
                      description="Analysis of the pattern of deepfake detection across frames"),
        ResponseSchema(name="technical_details", 
                      description="Technical details about the deepfake detection"),
        ResponseSchema(name="anomalies", 
                      description="Any notable anomalies or inconsistencies detected in the video"),
        ResponseSchema(name="recommendations", 
                      description="Recommendations for further analysis or actions based on findings")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    # Create a prompt template
    template = """
    You are a forensic video analyst specializing in deepfake detection. You've been given the results of 
    an automated video analysis that checks for signs of deepfake manipulation.

    Here's the data from the video analysis:
    {video_data}

    Based on this data, provide a comprehensive forensic report with the following sections:

    {format_instructions}

    Make sure to provide factual analysis based only on the data provided. Do not invent details.
    If the data is incomplete for any section, note this limitation.
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["video_data"],
        partial_variables={"format_instructions": format_instructions}
    )
    
    # Create the LLM chain using Groq instead of OpenAI
    llm = ChatGroq(
        model_name="llama3-70b-8192",  # Using Llama 3 70B model
        temperature=0
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run the chain
    try:
        result = chain.run(video_data=json.dumps(video_analysis_results, indent=2))
        parsed_output = output_parser.parse(result)
        return parsed_output
    except Exception as e:
        print(f"Error in LLM analysis: {e}")
        
        # Try direct Groq API as fallback
        try:
            client = Groq(api_key=os.environ["GROQ_API_KEY"])
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a forensic video analyst specializing in deepfake detection."
                    },
                    {
                        "role": "user",
                        "content": f"""Analyze these video deepfake detection results and provide a report with these sections:
                        1. Executive Summary (2-3 paragraphs)
                        2. Deepfake Assessment (likely fake or real, with confidence)
                        3. Frame Analysis (pattern analysis)
                        4. Technical Details
                        5. Anomalies Detected
                        6. Recommendations
                        
                        Here's the data:
                        {json.dumps(video_analysis_results, indent=2)}"""
                    }
                ],
                model="llama3-70b-8192",
                temperature=0
            )
            
            response_text = chat_completion.choices[0].message.content
            
            # Parse the response into our expected structure
            sections = {
                "executive_summary": "",
                "deepfake_assessment": "",
                "frame_analysis": "",
                "technical_details": "",
                "anomalies": "",
                "recommendations": ""
            }
            
            current_section = None
            lines = response_text.split('\n')
            
            for line in lines:
                line = line.strip()
                lower_line = line.lower()
                
                if "executive summary" in lower_line or "summary" in lower_line and not sections["executive_summary"]:
                    current_section = "executive_summary"
                    continue
                elif "assessment" in lower_line or "verdict" in lower_line:
                    current_section = "deepfake_assessment"
                    continue
                elif "frame analysis" in lower_line:
                    current_section = "frame_analysis"
                    continue
                elif "technical" in lower_line and "details" in lower_line:
                    current_section = "technical_details"
                    continue
                elif "anomal" in lower_line or "inconsistenc" in lower_line:
                    current_section = "anomalies"
                    continue
                elif "recommend" in lower_line:
                    current_section = "recommendations"
                    continue
                
                if current_section and line:
                    sections[current_section] += line + "\n"
            
            return sections
            
        except Exception as inner_e:
            print(f"Error in fallback LLM analysis: {inner_e}")
            # Return fallback report structure if LLM fails
            return {
                "executive_summary": "Error generating summary. Please check the raw analysis data.",
                "deepfake_assessment": "Assessment unavailable due to processing error.",
                "frame_analysis": "Frame analysis unavailable due to processing error.",
                "technical_details": "Technical details unavailable due to processing error.",
                "anomalies": "Anomaly detection unavailable due to processing error.",
                "recommendations": "Please review the raw analysis data manually."
            }

def generate_frame_xai_visualizations(frame_path, model_path):
    """
    Generate XAI visualizations (GradCAM and LIME) for a video frame
    
    Args:
        frame_path (str): Path to the frame image
        model_path (str): Path to the model checkpoint
        
    Returns:
        dict: Dictionary containing visualization data
    """
    try:
        from scripts.XAI.explanation.visualize import generate_xai_visualizations
        
        print(f"Generating XAI visualizations for frame: {frame_path}")
        print(f"Using model path: {model_path}")
        
        # Generate visualizations using the existing function
        visualizations = generate_xai_visualizations(
            frame_path, 
            model_path
        )
        
        if visualizations:
            print(f"Successfully generated visualizations. Available methods: {list(visualizations.keys())}")
        else:
            print("No visualizations were returned")
        
        return visualizations
    except Exception as e:
        print(f"Error generating XAI visualizations: {str(e)}")
        traceback_str = traceback.format_exc()
        print(f"Traceback: {traceback_str}")
        return None

def generate_video_pdf_report(output_path, case_number, investigator_name, video_analysis, 
                             video_path=None, sample_frames=None, model_path=None):
    """
    Generate a PDF report for video deepfake analysis with XAI visualizations
    
    Args:
        output_path (str): Path where to save the PDF report
        case_number (str): Unique case number for the investigation  
        investigator_name (str): Name of the investigator
        video_analysis (dict): Results from the video deepfake analysis
        video_path (str, optional): Path to the analyzed video file
        sample_frames (list, optional): List of paths to sample frames from the video
        model_path (str, optional): Path to the model checkpoint for XAI visualizations
    
    Returns:
        str: Path to the generated PDF report
    """
    # Ensure we have a case number
    if not case_number:
        case_number = generate_case_number()
    
    # Process the video analysis with LLM
    llm_analysis = analyze_with_llm(video_analysis)
    
    # Create document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=1.5*inch,
        bottomMargin=inch
    )
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Define custom styles
    custom_styles = {
        'CaseHeader': {
            'parent': 'Heading1',
            'fontSize': 16,
            'alignment': TA_CENTER,
            'spaceAfter': 12
        },
        'SectionTitle': {
            'parent': 'Heading2',
            'fontSize': 14,
            'spaceAfter': 6,
            'spaceBefore': 12,
            'textColor': colors.darkblue
        },
        'SubSectionTitle': {
            'parent': 'Heading3',
            'fontSize': 12,
            'spaceAfter': 6,
            'textColor': colors.darkblue
        },
        'Note': {
            'parent': 'Italic',
            'fontSize': 9,
            'textColor': colors.gray
        },
        'Verdict': {
            'parent': 'Normal',
            'fontSize': 12,
            'alignment': TA_CENTER,
            'textColor': colors.white,
            'backColor': colors.green,
            'borderPadding': 8,
            'borderWidth': 1,
            'borderColor': colors.black
        }
    }
    
    # Add the custom styles to the stylesheet
    for style_name, properties in custom_styles.items():
        parent_style = properties.pop('parent', 'Normal')
        styles.add(ParagraphStyle(name=style_name, parent=styles[parent_style], **properties))
    
    # Add VerdictReal as a variant of Verdict with green background
    styles.add(ParagraphStyle(name='VerdictReal', parent=styles['Verdict'], backColor=colors.green))
    
    # Modify existing styles
    styles['Normal'].fontSize = 10
    styles['Normal'].spaceBefore = 6
    styles['Normal'].leading = 14
    
    # Content elements list
    story = []
    
    # Title
    story.append(Paragraph(f"VIDEO DEEPFAKE ANALYSIS REPORT", styles['CaseHeader']))
    story.append(Spacer(1, 0.25*inch))
    
    # Case summary
    data = [
        ["Case Number:", case_number],
        ["Investigation Date:", datetime.datetime.now().strftime('%Y-%m-%d')],
        ["Investigator:", investigator_name],
        ["Analysis Method:", "Multi-frame Deepfake Detection & Forensic Analysis"]
    ]
    
    # Add video details if available
    if video_path:
        data.append(["Video Filename:", os.path.basename(video_path)])
    
    if "metadata" in video_analysis:
        if "duration" in video_analysis["metadata"]:
            data.append(["Video Duration:", f"{video_analysis['metadata']['duration']:.2f} seconds"])
        if "fps" in video_analysis["metadata"]:
            data.append(["Frame Rate:", f"{video_analysis['metadata']['fps']} FPS"])
        if "total_frames" in video_analysis["metadata"]:
            data.append(["Total Frames:", f"{video_analysis['metadata']['total_frames']}"])
        if "resolution" in video_analysis["metadata"]:
            data.append(["Resolution:", f"{video_analysis['metadata']['resolution']}"])
    
    case_table = Table(data, colWidths=[2*inch, 4*inch])
    case_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(case_table)
    story.append(Spacer(1, 0.25*inch))
    
    # Executive Summary
    story.append(Paragraph("EXECUTIVE SUMMARY", styles['SectionTitle']))
    summary_text = format_report_text(llm_analysis.get("executive_summary", "No summary available."))
    
    # Split by line breaks to create multiple paragraphs
    paragraphs = summary_text.split('\n')
    for para in paragraphs:
        if para.strip():  # Skip empty lines
            story.append(Paragraph(para.strip(), styles['Normal']))
    
    story.append(Spacer(1, 0.2*inch))
    
    # Deepfake Assessment
    assessment = llm_analysis.get("deepfake_assessment", "Assessment not available")
    
    # Determine if likely fake or real based on the assessment text
    is_fake = any(keyword in assessment.lower() for keyword in ["fake", "fabricated", "manipulated", "synthetic", "generated"])
    
    verdict_style = 'Verdict' if is_fake else 'VerdictReal'
    verdict_text = f"VERDICT: {assessment}"
    
    story.append(Paragraph(verdict_text, styles[verdict_style]))
    story.append(Spacer(1, 0.3*inch))
    
    # Add frame analysis chart if we have frame-by-frame scores
    if "frame_scores" in video_analysis:
        story.append(Paragraph("FRAME ANALYSIS", styles['SectionTitle']))
        story.append(Paragraph(format_report_text(llm_analysis.get("frame_analysis", "")), styles['Normal']))
        
        # Create and add the frame scores chart
        chart_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        create_frame_analysis_chart(video_analysis["frame_scores"], chart_path)
        
        img = RLImage(chart_path, width=6*inch, height=3*inch)
        story.append(img)
        story.append(Paragraph("Frame-by-frame deepfake probability scores (higher values indicate higher probability of manipulation)", styles['Note']))
        
        # Add overall average score
        if "average_score" in video_analysis:
            avg_score = video_analysis["average_score"]
            story.append(Paragraph(f"Average deepfake probability across all frames: <b>{avg_score:.2%}</b>", styles['Normal']))
        
        story.append(Spacer(1, 0.3*inch))
    
    # Add page break before technical details
    story.append(PageBreak())
    
    # Technical Details
    story.append(Paragraph("TECHNICAL DETAILS", styles['SectionTitle']))
    story.append(Paragraph(format_report_text(llm_analysis.get("technical_details", "No technical details available.")), styles['Normal']))
    story.append(Spacer(1, 0.3*inch))

    # === Add Grad-CAM and LIME Visualizations ===
    story.append(Paragraph("VISUAL EVIDENCE", styles['SectionTitle']))
    story.append(Spacer(1, 0.1*inch))

    lime_image = None
    gradcam_image = None
    png_paths = []  # For temporary PNG files

    # Process visualizations if available in video_analysis
    if "segmented" in video_analysis:
        segmented_data = video_analysis["segmented"]
        
        # Handle LIME visualization
        if "LIME" in segmented_data:
            viz_data = segmented_data["LIME"]
            for field in ["overlay", "saliency"]:
                if isinstance(viz_data, dict) and viz_data.get(field):
                    try:
                        import base64
                        base64_data = viz_data[field]
                        if base64_data.startswith('data:image'):
                            base64_data = base64_data.split(',', 1)[1]
                        binary_data = base64.b64decode(base64_data)
                        lime_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                        lime_file.write(binary_data)
                        lime_file.close()
                        lime_image = lime_file.name
                        break
                    except Exception as e:
                        print(f"Error processing LIME image: {e}")

        # Handle GradCAM++ visualization
        if "GradCAM++" in segmented_data:
            viz_data = segmented_data["GradCAM++"]
            for field in ["overlay", "saliency"]:
                if isinstance(viz_data, dict) and viz_data.get(field):
                    try:
                        import base64
                        base64_data = viz_data[field]
                        if base64_data.startswith('data:image'):
                            base64_data = base64_data.split(',', 1)[1]
                        binary_data = base64.b64decode(base64_data)
                        gradcam_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                        gradcam_file.write(binary_data)
                        gradcam_file.close()
                        gradcam_image = gradcam_file.name
                        break
                    except Exception as e:
                        print(f"Error processing GradCAM++ image: {e}")

    # Add LIME visualization
    if lime_image and os.path.exists(lime_image):
        try:
            img = Image.open(lime_image)
            aspect = img.height / img.width
            display_width = 5 * inch
            display_height = display_width * aspect
            
            if display_height > 4 * inch:
                display_height = 4 * inch
                display_width = display_height / aspect
                
            lime_png = lime_image.rsplit('.', 1)[0] + "_lime.png"
            img.convert('RGB').save(lime_png)
            png_paths.append(lime_png)
            
            story.append(Paragraph("LIME Visualization", styles['SubSectionTitle']))
            story.append(Paragraph("Key regions influencing the detection decision", styles['Note']))
            story.append(RLImage(lime_png, width=display_width, height=display_height))
            story.append(Spacer(1, 0.3*inch))
            
            # Add LIME explanation
            story.append(Paragraph("LIME Analysis Explanation:", styles['Normal']))
            story.append(Paragraph("""
            LIME (Local Interpretable Model-agnostic Explanations) highlights areas of the image that
            most strongly influenced the AI's decision about deepfake detection. The highlighted regions
            show which parts of the image were most important in determining whether the content is manipulated.
            Areas with strong highlights typically indicate problematic regions where visual artifacts,
            inconsistent textures, or unnatural features were detected.
            """, styles['Normal']))
        except Exception as e:
            story.append(Paragraph(f"Error loading LIME: {str(e)}", styles['Note']))

    # Add GradCAM++ visualization
    if gradcam_image and os.path.exists(gradcam_image):
        try:
            img = Image.open(gradcam_image)
            aspect = img.height / img.width
            display_width = 5 * inch
            display_height = display_width * aspect
            
            if display_height > 4 * inch:
                display_height = 4 * inch
                display_width = display_height / aspect
                
            gradcam_png = gradcam_image.rsplit('.', 1)[0] + "_gradcam.png"
            img.convert('RGB').save(gradcam_png)
            png_paths.append(gradcam_png)
            
            story.append(Paragraph("GradCAM++ Visualization", styles['SubSectionTitle']))
            story.append(Paragraph("Areas indicating potential manipulation", styles['Note']))
            story.append(RLImage(gradcam_png, width=display_width, height=display_height))
            story.append(Spacer(1, 0.3*inch))
            
            # Add GradCAM explanation
            story.append(Paragraph("GradCAM++ Analysis Explanation:", styles['Normal']))
            story.append(Paragraph("""
            Gradient-weighted Class Activation Mapping (GradCAM++) visualizes which regions of the image
            activated the deepfake detection neural network most strongly. Warmer colors (red/yellow) 
            indicate areas that strongly contribute to the detection of manipulated content. This 
            technique helps reveal what the AI "sees" when it detects signs of digital manipulation.
            """, styles['Normal']))
        except Exception as e:
            story.append(Paragraph(f"Error loading GradCAM++: {str(e)}", styles['Note']))

    # Add lip sync analysis explanation if available
    if "lip_sync_analysis" in video_analysis:
        lip_sync = video_analysis["lip_sync_analysis"]
        
        # Add descriptive text about lip sync analysis
        story.append(Paragraph("Lip Synchronization Analysis", styles['SubSectionTitle']))
        story.append(Paragraph(
            "Lip synchronization analysis examines the correlation between audio speech patterns and visual lip movements. "
            "Deepfakes often show discrepancies in this correlation, as generating perfectly synchronized lip movements "
            "is one of the most challenging aspects of creating convincing fake videos.",
            styles['Normal']
        ))
        
        # Add specific lip sync results if available
        if isinstance(lip_sync, dict):
            lip_sync_data = []
            
            if "real_probability" in lip_sync:
                real_prob = lip_sync["real_probability"]
                lip_sync_data.append(["Real Probability", f"{real_prob:.3f} ({real_prob*100:.1f}%)"])
                
            if "fake_probability" in lip_sync:
                fake_prob = lip_sync["fake_probability"]
                lip_sync_data.append(["Fake Probability", f"{fake_prob:.3f} ({fake_prob*100:.1f}%)"])
                
            if "description" in lip_sync:
                lip_sync_data.append(["Analysis", lip_sync["description"]])
                
            if lip_sync_data:
                table = Table(lip_sync_data, colWidths=[2*inch, 4*inch])
                table.setStyle(TableStyle([
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                    ('PADDING', (0, 0), (-1, -1), 6),
                ]))
                story.append(table)
        elif isinstance(lip_sync, (int, float)):
            # If lip_sync is just a probability value
            story.append(Paragraph(f"Real Voice Probability: {lip_sync:.3f} ({lip_sync*100:.1f}%)", styles['Normal']))
            story.append(Paragraph(f"Fake Voice Probability: {1-lip_sync:.3f} ({(1-lip_sync)*100:.1f}%)", styles['Normal']))
        
        story.append(Spacer(1, 0.3*inch))

    # Sample frames section with XAI visualizations
    if sample_frames and len(sample_frames) > 0:
        story.append(Paragraph("SAMPLE FRAMES ANALYZED", styles['SectionTitle']))
        story.append(Paragraph("Representative frames extracted from the video for analysis:", styles['Normal']))
        
        # Display sample frames (maximum 3 per page)
        for i, frame_path in enumerate(sample_frames[:6]):  # Limit to 6 sample frames
            try:
                img = Image.open(frame_path)
                img_width, img_height = img.size
                aspect = img_height / float(img_width)
                
                # Calculate image display size
                display_width = 3 * inch  # Smaller than full width
                display_height = display_width * aspect
                
                # Ensure height doesn't exceed a reasonable size
                if display_height > 2.5 * inch:
                    display_height = 2.5 * inch
                    display_width = display_height / aspect
                
                # Add frame image with caption
                frame_img = RLImage(frame_path, width=display_width, height=display_height)
                story.append(frame_img)
                
                # Add frame information
                frame_num = f"Frame #{i+1}"
                
                # Add deepfake score if available for this specific frame
                if "frame_scores" in video_analysis and i < len(video_analysis["frame_scores"]):
                    score = video_analysis["frame_scores"][i]
                    frame_num += f" - Deepfake probability: {score:.2%}"
                
                story.append(Paragraph(frame_num, styles['Note']))
                story.append(Spacer(1, 0.2*inch))
                
                # Add XAI visualizations for high-score frames
                if model_path and "frame_scores" in video_analysis and i < len(video_analysis["frame_scores"]):
                    score = video_analysis["frame_scores"][i]
                    # Only generate visualizations for frames with high deepfake probability (above 0.6)
                    if score > 0.6:
                        # Add a note that we're generating XAI visualizations
                        story.append(Paragraph("Generating XAI visualizations for this high-probability frame...", styles['Note']))
                        
                        # Generate XAI visualizations
                        xai_results = generate_frame_xai_visualizations(frame_path, model_path)
                        
                        if xai_results:
                            # Add page break for XAI results
                            story.append(PageBreak())
                            story.append(Paragraph(f"EXPLAINABLE AI ANALYSIS FOR FRAME #{i+1}", styles['SectionTitle']))
                            
                            # Add GradCAM visualization if available
                            if 'GradCAM++' in xai_results and 'overlay' in xai_results['GradCAM++']:
                                story.append(Paragraph("GradCAM++ Visualization", styles['SubSectionTitle']))
                                story.append(Paragraph("GradCAM++ highlights regions that most activated the deepfake detection model. Warmer colors (red/yellow) indicate areas with stronger influence on the model's decision.", styles['Normal']))
                                
                                gradcam_path = xai_results['GradCAM++']['overlay']
                                gradcam_img = RLImage(gradcam_path, width=display_width*1.2, height=display_height*1.2)
                                story.append(gradcam_img)
                                story.append(Spacer(1, 0.2*inch))
                            
                            # Add LIME visualization if available
                            if 'LIME' in xai_results and 'overlay' in xai_results['LIME']:
                                story.append(Paragraph("LIME Visualization", styles['SubSectionTitle']))
                                story.append(Paragraph("LIME (Local Interpretable Model-agnostic Explanations) shows which parts of the image influenced the model's classification decision. Highlighted areas contributed most significantly to the deepfake detection.", styles['Normal']))
                                
                                lime_path = xai_results['LIME']['overlay']
                                lime_img = RLImage(lime_path, width=display_width*1.2, height=display_height*1.2)
                                story.append(lime_img)
                                story.append(Spacer(1, 0.2*inch))
                            
                            # Add explanation of what these visualizations mean
                            story.append(Paragraph("Interpretation of XAI Visualizations", styles['SubSectionTitle']))
                            story.append(Paragraph("""
                            The highlighted regions in the visualizations above indicate areas that the AI model focused on when determining if the frame is manipulated. 
                            Common indicators of deepfakes include:
                            
                            • Inconsistent facial features or skin textures
                            • Unnatural borders around modified regions
                            • Inconsistent lighting or shadows
                            • Artifacts around eyes, mouth, or hair
                            • Blending inconsistencies between manipulated and original content
                            
                            Areas with strong activations (bright regions in GradCAM++ or highlighted regions in LIME) that correspond with these common indicators strengthen the evidence of manipulation.
                            """, styles['Normal']))
                
                # Add page break after every 3 frames
                if (i+1) % 3 == 0 and i < len(sample_frames)-1:
                    story.append(PageBreak())
                    story.append(Paragraph("SAMPLE FRAMES (CONTINUED)", styles['SectionTitle']))
                
            except Exception as e:
                story.append(Paragraph(f"Error loading frame image: {str(e)}", styles['Note']))
                
    # Anomalies Section
    story.append(Paragraph("ANOMALIES DETECTED", styles['SectionTitle']))
    story.append(Paragraph(format_report_text(llm_analysis.get("anomalies", "No specific anomalies detected.")), styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Recommendations Section
    story.append(Paragraph("RECOMMENDATIONS", styles['SectionTitle']))
    story.append(Paragraph(format_report_text(llm_analysis.get("recommendations", "No specific recommendations available.")), styles['Normal']))

    doc.build(story, onFirstPage=lambda canvas, doc: create_header_footer(canvas, doc, case_number, investigator_name),
             onLaterPages=lambda canvas, doc: create_header_footer(canvas, doc, case_number, investigator_name))
    
    # Clean up temporary files
    if "frame_scores" in video_analysis and 'chart_path' in locals() and os.path.exists(chart_path):
        os.unlink(chart_path)
    for tmp in [lime_image, gradcam_image] + png_paths:
        if tmp and os.path.exists(tmp):
            try: os.unlink(tmp)
            except: pass

    return output_path

# Example usage code for main.py
"""
from scripts.VideoReport import generate_video_pdf_report, generate_case_number

@app.route('/generate_video_report', methods=['POST'])
def handle_generate_video_report():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data received"}), 400
            
        # Extract required information
        video_analysis = data.get("analysis_results", {})
        investigator_name = data.get("investigator_name", "AI Detection System")
        case_number = data.get("case_number", generate_case_number())
        
        # Create a unique filename for the report
        filename = f"video_report_{case_number}.pdf"
        report_path = os.path.join(REPORTS_DIR, filename)
        
        # Extract sample frames if provided
        sample_frames = data.get("sample_frames", [])
        video_path = data.get("video_path", None)
        
        # Generate the report
        generate_video_pdf_report(
            report_path, 
            case_number, 
            investigator_name, 
            video_analysis,
            video_path,
            sample_frames
        )
        
        # Return the report
        return send_file(report_path, as_attachment=True)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
"""

# Test execution
if __name__ == '__main__':
    # Sample video analysis results
    sample_analysis = {
        "metadata": {
            "filename": "sample_video.mp4",
            "duration": 10.5,
            "fps": 30,
            "total_frames": 315,
            "resolution": "1280x720"
        },
        "frame_scores": [0.12, 0.15, 0.18, 0.25, 0.45, 0.78, 0.82, 0.85, 0.79, 0.72],
        "average_score": 0.511,
        "highest_score": 0.85,
        "frame_with_highest_score": 8,
        "detection_threshold": 0.5,
        "lip_sync_analysis": {
            "real_probability": 0.23,
            "fake_probability": 0.77,
            "description": "Significant lip sync inconsistencies detected"
        }
    }
    
    # Generate report
    report_path = os.path.join(os.path.dirname(__file__), "sample_video_report.pdf")
    generate_video_pdf_report(
        report_path,
        generate_case_number(),
        "AI Detection System",
        sample_analysis
    )
    
    print(f"Report generated at: {report_path}")