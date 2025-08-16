import os
import io
import uuid
import json
import datetime
import tempfile
import time
import shutil
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.platypus import PageBreak, KeepTogether, HRFlowable, Flowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing
import sys
from PIL.ExifTags import TAGS, GPSTAGS
import c2pa
import hashlib

def generate_case_number_with_checksum():
    """Generate a unique case number with date prefix and SHA-256 checksum"""
    date_prefix = datetime.datetime.now().strftime("%Y%m%d")
    unique_id = str(uuid.uuid4().hex)[:6].upper()
    case_number = f"{date_prefix}-{unique_id}"
    
    # Generate SHA-256 checksum of the case number
    checksum = hashlib.sha256(case_number.encode()).hexdigest()[:8]
    
    return case_number, checksum, "SHA-256"
# Import exiftool directly rather than from a module
try:
    import exiftool
except ImportError:
    print("PyExifTool not installed. Install with: pip install pyexiftool")
    # Create a fallback function
    def get_exif_fallback(image_path, return_data=False):
        """Fallback function when exiftool is not available"""
        try:
            with Image.open(image_path) as img:
                exif_data = {}
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    if exif:
                        exif_data = {"EXIF": exif}
                if return_data:
                    return exif_data
                else:
                    print(exif_data)
        except Exception as e:
            if return_data:
                return {"Error": str(e)}
            else:
                print(f"Error extracting EXIF data: {str(e)}")

try:
    import c2pa
except ImportError:
    c2pa = None

# Add this function (modified version of your getc2pa)
def get_c2pa_data(image_path):
    """Extract C2PA metadata if available"""
    c2pa_data = {}
    try:
        if c2pa:
            # Create a reader from the file path
            reader = c2pa.Reader.from_file(image_path)
            if reader.manifest_store:
                # Convert to dict and clean up binary data
                c2pa_data = json.loads(reader.json())
                # Remove large binary data that can't be serialized
                c2pa_data.pop('signature', None)
                c2pa_data.pop('cert_chain', None)
            else:
                c2pa_data = {"status": "No C2PA manifest found"}
        else:
            c2pa_data = {"error": "c2pa library not installed - install with: pip install c2pa"}
    except Exception as err:
        c2pa_data = {"error": str(err)}
    return c2pa_data

def get_pil_exif(image_path):
    """Extract EXIF metadata using PIL with detailed tag processing"""
    exif_data = {}
    try:
        with Image.open(image_path) as img:
            if hasattr(img, '_getexif') and img._getexif() is not None:
                exif_info = img._getexif()
                # Process standard EXIF tags
                for tag_id, value in exif_info.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    exif_data[tag_name] = value

                # Process GPS information separately
                if 'GPSInfo' in exif_data:
                    gps_data = {}
                    for gps_tag in exif_data['GPSInfo'].keys():
                        gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                        gps_data[gps_tag_name] = exif_data['GPSInfo'][gps_tag]
                    exif_data['GPSInfo'] = gps_data

                # Extract additional image information
                exif_data['Image Information'] = {
                    'Format': img.format,
                    'Mode': img.mode,
                    'Size': img.size,
                    'Width': img.width,
                    'Height': img.height,
                    'Animated': getattr(img, 'is_animated', False),
                    'Frames': getattr(img, 'n_frames', 1)
                }

    except Exception as e:
        exif_data['Error'] = f"EXIF extraction error: {str(e)}"
    
    return exif_data

class HeatmapImage(Flowable):
    """A flowable for adding heatmap images with captions"""
    def __init__(self, image_path, width=6*inch, height=None, caption=""):
        Flowable.__init__(self)
        self.img = Image.open(image_path)
        self.caption = caption
        self.width = width
        
        # Calculate height based on aspect ratio if not provided
        if height is None:
            self.height = width * self.img.height / self.img.width
        else:
            self.height = height
            
    def draw(self):
        # Draw image
        self.canv.drawImage(self.img, 0, 12, width=self.width, height=self.height, 
                           preserveAspectRatio=True)
        
        # Draw caption
        if self.caption:
            self.canv.setFont("Helvetica", 9)
            self.canv.drawCentredString(self.width/2, 0, self.caption)

def generate_case_number():
    """Generate a unique case number with date prefix"""
    date_prefix = datetime.datetime.now().strftime("%Y%m%d")
    unique_id = str(uuid.uuid4().hex)[:6].upper()
    return f"{date_prefix}-{unique_id}"

def format_metadata_table(metadata):
    """Format the metadata as a table for the report"""
    if not metadata:
        return [["No metadata available", ""]]
        
    formatted_data = []
    
    # Handle different input formats
    if isinstance(metadata, str):
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            return [["Raw Metadata", metadata]]
    else:
        metadata_dict = metadata
    
    # Flatten nested dictionaries with dot notation
    def flatten_dict(d, prefix=""):
        items = []
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, key).items())
            elif isinstance(v, list) and all(isinstance(x, dict) for x in v):
                # Handle lists of dictionaries
                for i, item in enumerate(v):
                    items.extend(flatten_dict(item, f"{key}[{i}]").items())
            else:
                items.append((key, v))
        return dict(items)
    
    # Flatten if dict
    if isinstance(metadata_dict, dict):
        flat_dict = flatten_dict(metadata_dict)
        formatted_data = [[k, str(v)] for k, v in flat_dict.items()]
    else:
        formatted_data = [["Raw Metadata", str(metadata_dict)]]
    
    return formatted_data

def create_header_footer(canvas, doc, case_number, investigator_name,checksum=None, checksum_algorithm=None):
    """Create header and footer for each page"""
    canvas.saveState()
    
    # Header
    canvas.setFont('Helvetica-Bold', 10)
    canvas.drawString(inch, 10.5 * inch, f"CASE #{case_number}")
    # Add checksum information if provided
    if checksum and checksum_algorithm:
        canvas.setFont('Helvetica', 8)
        canvas.drawString(inch, 10.35 * inch, f"Checksum ({checksum_algorithm}): {checksum}")
    canvas.setFont('Helvetica', 9)
    canvas.drawString(inch, 10.3 * inch, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    canvas.drawString(inch, 10.1 * inch, f"Investigator: {investigator_name}")
    
    # Add a line under the header
    canvas.line(inch, 10*inch, 7.5*inch, 10*inch)
    
    # Footer
    canvas.setFont('Helvetica', 8)
    canvas.line(inch, 0.75*inch, 7.5*inch, 0.75*inch)
    canvas.drawString(inch, 0.5 * inch, "CONFIDENTIAL - FORENSIC INVESTIGATION REPORT")
    canvas.drawRightString(7.5 * inch, 0.5 * inch, f"Page {doc.page}")
    
    canvas.restoreState()

def get_exif_data(image_path, return_data=False):
    """
    Extracts EXIF metadata from the given image file.
    
    Args:
        image_path (str): Path to the image file
        return_data (bool): If True, returns the data instead of printing it
        
    Returns:
        dict: EXIF metadata if return_data is True, None otherwise
    """
    try:
        # Check if exiftool module is available
        if 'exiftool' in sys.modules:
            with exiftool.ExifTool() as et:
                metadata = et.execute_json("-j", image_path)  # Use execute_json() to get structured data
                
            if metadata:
                if return_data:
                    return metadata[0]  # Return the first item from the list
                else:
                    print(json.dumps(metadata[0], indent=4))  # Pretty print JSON output
            else:
                if return_data:
                    return {}
                else:
                    print("No EXIF data found.")
        else:
            # Use fallback if exiftool module is not available
            return get_exif_fallback(image_path, return_data)
                
    except Exception as e:
        if return_data:
            return {"Error": str(e)}
        else:
            print(f"Error extracting EXIF data: {str(e)}")
    
    if return_data:
        return {}

def format_report_text(text):
    """Format text with basic markdown parsing for better PDF rendering"""
    import re
    
    # Replace markdown bold with ReportLab bold tags
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Replace markdown italic with ReportLab italic tags
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    
    # Handle numbered lists by adding proper spacing
    text = re.sub(r'(\d+)\.\s+', r'<bullet>\1.</bullet> ', text)
    
    return text

def generate_pdf_report(output_path, case_number, investigator_name, analysis_results, image_path, grad_cam_path=None):
    """Generate a PDF report for deepfake analysis"""
    
    # Ensure we have a case number
    if not case_number:
        case_number, checksum, checksum_algorithm = generate_case_number_with_checksum()
    else:
        # Generate checksum for provided case number
        checksum = hashlib.sha256(case_number.encode()).hexdigest()[:8]
        checksum_algorithm = "SHA-256"
    
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
    
    # Define custom styles in a more consolidated way
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
        'CellText': {
            'parent': 'Normal',
            'fontSize': 9
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
            'backColor': colors.red,
            'borderPadding': 8,
            'borderWidth': 1,
            'borderColor': colors.black,
            'borderRadius': 8
        }
    }
    
    # Add the custom styles to the stylesheet
    for style_name, properties in custom_styles.items():
        parent_style = properties.pop('parent', 'Normal')
        styles.add(ParagraphStyle(name=style_name, parent=styles[parent_style], **properties))
    
    # Add VerdictReal as a variant of Verdict
    styles.add(ParagraphStyle(name='VerdictReal', parent=styles['Verdict'], backColor=colors.green))
    
    # Modify existing styles
    styles['Normal'].fontSize = 10
    styles['Normal'].spaceBefore = 6
    styles['Normal'].leading = 14
    styles["Code"].fontSize = 8
    styles["Code"].leading = 10
    
    # Story (content elements)
    story = []
    
    # Title
    story.append(Paragraph(f"DEEPFAKE ANALYSIS REPORT", styles['CaseHeader']))
    story.append(Spacer(1, 0.25*inch))
    
    # Case summary
    data = [
        ["Case Number:", case_number],
        ["Case Checksum:", f"{checksum} ({checksum_algorithm})"],
        ["Investigation Date:", datetime.datetime.now().strftime('%Y-%m-%d')],
        ["Investigator:", investigator_name],
        ["Analysis Method:", "Multi-modal Deepfake Detection & Forensic Analysis"]
    ]
    
    case_table = Table(data, colWidths=[2*inch, 4*inch])
    case_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(case_table)
    story.append(Spacer(1, 0.25*inch))
    
    # Image Preview
    try:
        story.append(Paragraph("MEDIA ANALYZED", styles['SectionTitle']))
        img = Image.open(image_path)
        img_width, img_height = img.size
        aspect = img_height / float(img_width)
        
        # Calculate image display size, keeping aspect ratio
        display_width = 5 * inch
        display_height = display_width * aspect
        
        # Ensure height doesn't exceed page height
        if display_height > 4 * inch:
            display_height = 4 * inch
            display_width = display_height / aspect
            
        img_obj = RLImage(image_path, width=display_width, height=display_height)
        story.append(img_obj)
        story.append(Paragraph(f"Filename: {os.path.basename(image_path)}<br/>Dimensions: {img_width}x{img_height} pixels", styles['Note']))
        story.append(Spacer(1, 0.25*inch))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    except Exception as e:
        story.append(Paragraph(f"Error loading image: {str(e)}", styles['Note']))

    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))

        # Add EXIF Metadata Section
    story.append(Paragraph("DIGITAL FORENSICS METADATA", styles['SectionTitle']))
        
        # Extract EXIF data
    exif_metadata = get_pil_exif(image_path)
        
    if exif_metadata:
            # Create metadata table
        table_data = [["Metadata Field", "Value"]]
            
            # Process nested data using existing format_metadata_table
        formatted_metadata = format_metadata_table(exif_metadata)
        table_data.extend(formatted_metadata)
            
            # Create table with alternating colors
        metadata_table = Table(table_data, colWidths=[2.5*inch, 4.5*inch])
        metadata_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.black),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 12),
                ('BACKGROUND', (0,1), (-1,-1), colors.white),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white])
            ]))
            
        story.append(KeepTogether([
                Paragraph("Technical Metadata Extraction", styles['SubSectionTitle']),
                metadata_table,
                Spacer(1, 0.25*inch)
        ]))
    else:
            story.append(Paragraph("No technical metadata could be extracted", styles['Note']))
            
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))

    story.append(Paragraph("CONTENT CREDENTIALS (C2PA)", styles['SectionTitle']))
        
        # Get C2PA data
    c2pa_data = get_c2pa_data(image_path)
        
    if c2pa_data:
            # Create C2PA table
        table_data = [["C2PA Field", "Value"]]
        formatted_c2pa = format_metadata_table(c2pa_data)
        table_data.extend(formatted_c2pa)
            
            # Create table with matching style
        c2pa_table = Table(table_data, colWidths=[2.5*inch, 4.5*inch])
        c2pa_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.black),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 12),
                ('BACKGROUND', (0,1), (-1,-1), colors.white),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white])
            ]))
            
        story.append(KeepTogether([
                Paragraph("Content Authenticity Verification", styles['SubSectionTitle']),
                c2pa_table,
                Spacer(1, 0.25*inch)
            ]))
            
            # Add validation summary
        if 'error' not in c2pa_data:
                validation_status = "Valid C2PA Signature Found" if c2pa_data.get('signature_valid', False) else "Invalid or Missing C2PA Signature"
                story.append(Paragraph(f"Validation Status: {validation_status}", styles['Note']))
        else:
            story.append(Paragraph("No C2PA content credentials found", styles['Note']))
            
        story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    
    # Analysis Results
    story.append(Paragraph("ANALYSIS RESULTS", styles['SectionTitle']))
    
    # Extract the relevant parts from the analysis_results
    if isinstance(analysis_results, str):
        try:
            results_dict = json.loads(analysis_results)
        except json.JSONDecodeError:
            results_dict = {"Error": "Invalid JSON format in analysis results"}
    else:
        results_dict = analysis_results
    
    # Add summary verdict first (most important)
    if "deepfake" in results_dict:
        verdict = results_dict["deepfake"]
        is_fake = "Fake" in verdict
        
        verdict_style = 'Verdict' if is_fake else 'VerdictReal'
        verdict_text = f"VERDICT: {verdict}"
        
        story.append(Paragraph(verdict_text, styles[verdict_style]))
        story.append(Spacer(1, 0.2*inch))
    
    # Add report info if available
    if "report" in results_dict:
        report_data = results_dict["report"]
        
        # Final Summary if available
        if "Final Summary and Verdict" in report_data:
            story.append(Paragraph("Final Analysis", styles['SubSectionTitle']))
            summary_text = format_report_text(report_data["Final Summary and Verdict"])
            
            # Split by line breaks to create multiple paragraphs
            paragraphs = summary_text.split('\n')
            for para in paragraphs:
                if para.strip():  # Skip empty lines
                    story.append(Paragraph(para.strip(), styles['Normal']))
            
            story.append(Spacer(1, 0.2*inch))
        
        # Visual Content Analysis
        if "Visual Content Analysis" in report_data:
            story.append(Paragraph("Visual Content Analysis", styles['SubSectionTitle']))
            story.append(Paragraph(report_data["Visual Content Analysis"], styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Anomaly Detection
        if "Anomaly Detection" in report_data:
            story.append(Paragraph("Anomaly Detection", styles['SubSectionTitle']))
            story.append(Paragraph(report_data["Anomaly Detection"], styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Text Extraction if available
        if "Text Extraction" in report_data and report_data["Text Extraction"].strip() != "No text could be detected in the image after multiple OCR attempts.":
            story.append(Paragraph("Text Extracted from Media", styles['SubSectionTitle']))
            story.append(Paragraph(report_data["Text Extraction"], styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Deep Learning Model Evaluation
        if "Deep Learning Model Evaluation" in report_data:
            story.append(Paragraph("AI Model Evaluation", styles['SubSectionTitle']))
            story.append(Paragraph(report_data["Deep Learning Model Evaluation"], styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
    else:
        # Fall back to old format
        # Final Summary if available
        if "Final Summary and Verdict" in results_dict:
            story.append(Paragraph("Final Analysis", styles['SubSectionTitle']))
            story.append(Paragraph(results_dict["Final Summary and Verdict"], styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Visual Content Analysis
        if "Visual Content Analysis" in results_dict:
            story.append(Paragraph("Visual Content Analysis", styles['SubSectionTitle']))
            story.append(Paragraph(results_dict["Visual Content Analysis"], styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Anomaly Detection
        if "Anomaly Detection" in results_dict:
            story.append(Paragraph("Anomaly Detection", styles['SubSectionTitle']))
            story.append(Paragraph(results_dict["Anomaly Detection"], styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
    
    # Add page break before technical details
    story.append(PageBreak())
    
    # Technical Details
    story.append(Paragraph("TECHNICAL DETAILS", styles['SectionTitle']))
    
    # Lip Sync Analysis (for videos)
    if "lip_sync_analysis" in results_dict:
        story.append(Paragraph("Lip Synchronization Analysis", styles['SubSectionTitle']))
        lip_sync = results_dict["lip_sync_analysis"]
        
        if isinstance(lip_sync, dict):
            if "real_probability" in lip_sync and "fake_probability" in lip_sync:
                real_prob = lip_sync["real_probability"]
                fake_prob = lip_sync["fake_probability"]
                
                data = [
                    ["Real Probability", f"{real_prob:.3f} ({real_prob*100:.1f}%)"],
                    ["Fake Probability", f"{fake_prob:.3f} ({fake_prob*100:.1f}%)"]
                ]
                
                if "description" in lip_sync:
                    data.append(["Analysis", lip_sync["description"]])
                
                if "processing_time_seconds" in lip_sync:
                    data.append(["Processing Time", f"{lip_sync['processing_time_seconds']:.2f} seconds"])
                
                table = Table(data, colWidths=[2*inch, 4*inch])
                table.setStyle(TableStyle([
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                    ('PADDING', (0, 0), (-1, -1), 6),
                ]))
                story.append(table)
            elif isinstance(lip_sync, (int, float, str)):
                story.append(Paragraph(f"Result: {lip_sync}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
    
    # Grad-CAM and LIME Visualizations
    story.append(Paragraph("Visual Evidence", styles['SubSectionTitle']))
    story.append(Spacer(1, 0.1*inch))

    lime_image = None
    gradcam_image = None
    png_paths = []  # Keep track of PNG paths for cleanup

    if "segmented" in results_dict:
        segmented_data = results_dict["segmented"]
        if isinstance(segmented_data, dict):
            # Check for LIME visualization
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
                            print(f"Successfully created temporary LIME visualization file: {lime_image}")
                            break
                        except Exception as e:
                            print(f"Error processing LIME {field} image: {e}")
            
            # Check for GradCAM++ visualization
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
                            print(f"Successfully created temporary GradCAM++ visualization file: {gradcam_image}")
                            break
                        except Exception as e:
                            print(f"Error processing GradCAM++ {field} image: {e}")

    # Add LIME visualization first if available
    if lime_image and os.path.exists(lime_image):
        try:
            img = Image.open(lime_image)
            img_width, img_height = img.size
            aspect = img_height / float(img_width)
            display_width = 5 * inch
            display_height = display_width * aspect
            if display_height > 4 * inch:
                display_height = 4 * inch
                display_width = display_height / aspect
            if img.mode not in ['RGB', 'RGBA']:
                img = img.convert('RGB')
            
            # Use a unique filename for the PNG
            lime_png_path = lime_image.rsplit('.', 1)[0] + "_lime.png"
            img.save(lime_png_path, "PNG")
            png_paths.append(lime_png_path)
            print(f"Saved LIME visualization as PNG: {lime_png_path}")
            
            # Add a clear title and the image
            story.append(Paragraph("LIME Visualization", styles['SubSectionTitle']))
            story.append(Paragraph("Areas highlighted by LIME algorithm showing regions that influenced the detection decision", styles['Note']))
            img_obj = RLImage(lime_png_path, width=display_width, height=display_height)
            story.append(img_obj)
            
            # Add significant space after this visualization
            story.append(Spacer(1, 0.5*inch))
        except Exception as e:
            story.append(Paragraph(f"Error loading LIME visualization image: {str(e)}", styles['Note']))
            print(f"Exception when adding LIME visualization to PDF: {e}")

    # Add GradCAM++ visualization if available
    if gradcam_image and os.path.exists(gradcam_image):
        try:
            img = Image.open(gradcam_image)
            img_width, img_height = img.size
            aspect = img_height / float(img_width)
            display_width = 5 * inch
            display_height = display_width * aspect
            if display_height > 4 * inch:
                display_height = 4 * inch
                display_width = display_height / aspect
            if img.mode not in ['RGB', 'RGBA']:
                img = img.convert('RGB')
            
            # Use a unique filename for the PNG
            gradcam_png_path = gradcam_image.rsplit('.', 1)[0] + "_gradcam.png"
            img.save(gradcam_png_path, "PNG")
            png_paths.append(gradcam_png_path)
            print(f"Saved GradCAM++ visualization as PNG: {gradcam_png_path}")
            
            # Consider adding a page break if the images are large
            if display_height > 3 * inch:
                story.append(PageBreak())
                
            # Add a clear title and the image
            story.append(Paragraph("GradCAM++ Visualization", styles['SubSectionTitle']))
            story.append(Paragraph("Areas highlighted by GradCAM++ algorithm showing regions of potential manipulation", styles['Note']))
            img_obj = RLImage(gradcam_png_path, width=display_width, height=display_height)
            story.append(img_obj)
        except Exception as e:
            story.append(Paragraph(f"Error loading GradCAM++ visualization image: {str(e)}", styles['Note']))
            print(f"Exception when adding GradCAM++ visualization to PDF: {e}")

    # Continue with PDF generation...
    doc.build(story, onFirstPage=lambda canvas, doc: create_header_footer(canvas, doc, case_number, investigator_name),
             onLaterPages=lambda canvas, doc: create_header_footer(canvas, doc, case_number, investigator_name))

    # Clean up temporary files after PDF generation
    for tmp in [lime_image, gradcam_image] + png_paths:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
                print(f"Cleaned up temporary file: {tmp}")
            except Exception as e:
                print(f"Error cleaning up temporary file {tmp}: {e}")
    
    return output_path