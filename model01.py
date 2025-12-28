"""
Streamlit application for extracting MCQs and Diagrams from PDFs. Uses Gemini 2.5 Pro for text analysis and Gemini 2.5 Flash Image for generative diagram reproduction. Includes PDF-to-image conversion and automatic retries for API rate limits.
"""
import streamlit as st
import time
import os
import json
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Tuple, Union
import random
import fitz # PyMuPDF


st.set_page_config(page_title="PDF MCQ & Generative Diagram Extractor", layout="wide")

# --- 1. Pydantic Schemas ---

class QuestionData(BaseModel):
    id: int
    chapter: str = Field(default="Academic")
    question: str
    question_latex: str
    page_index: int = 0
    diagram_bboxes: List[List[int]] = [] # Loose boxes from full page
    extracted_image_names: List[str] = []

class KeyData(BaseModel):
    id: int
    answer_option: str

class SolutionData(BaseModel):
    id: int
    solution: str

class IntermediateResult(BaseModel):
    questions: List[QuestionData] = []
    answer_keys: List[KeyData] = []
    solutions: List[SolutionData] = []

# --- 2. System Instructions ---

# Pass 1: For the Text Model (Discovery)
SYSTEM_INSTRUCTION = """
Analyze the exam page. Extract Questions, Answers, and Solutions.
--- PASS 1: LOOSE GROUNDING ---
Identify every technical diagram. Provide a LOOSE bounding box [ymin, xmin, ymax, xmax].
Capture the diagram and a surrounding margin to ensure no labels are missed.
"""

GENERATIVE_PROMPT = """### ROLE: 
Expert Document Vision Engineer & Geometric Analyst.

### TASK:
Analyze the input image with sub-pixel attention to identify the absolute spatial boundaries of the primary technical diagram.

### EXTRACTION LOGIC:
1. SEMANTIC FILTERING: Distinguish between "Question Text" (prose/paragraphs/problem statements) and "Diagrammatic Text" (labels, variables, and values like h1, u1, or 2.66 m/s²).
2. VECTOR BOUNDS: Find the extreme outer edges of all geometric primitives:
    - Vertical: From the highest anchor point (top horizontal support line) to the lowest horizontal boundary.
    - Horizontal: From the leftmost tip of the 'h2' bracket to the rightmost character of the 'm/s²' label.
3. EDGE BUFFERING: Do not truncate subscripts (like the '2' in m/s²) or radicals (square root symbols). Include a strictly tight 2% padding around these detected edges.

### CONSTRAINTS:
- Exclude all surrounding whitespace that contains paragraph fragments.
- Exclude page headers, footers, or watermarks.
- If a label is logically tied to a bracket (like 'h2'), its coordinates MUST be included in the box.

### OUTPUT SPECIFICATION:
Your response must be ONLY a valid JSON object. Do not include conversational text or markdown code blocks.
Normalize coordinates to 1000x1000 scale where [0,0,0,0] is top-left and [1000,1000,1000,1000] is bottom-right.

Output format:
[{"box_2d": [ymin, xmin, ymax, xmax], "label": "technical_diagram"}]"""


def call_gemini_with_retry(client, model_name, contents, config=None, retries=5, initial_delay=4, response_schema=None):
    """
    Wraps the API call with exponential backoff for 429 errors.
    Also handles ValidationErrors if response_schema is provided.
    """
    for attempt in range(retries):
        try:
            # 1. Make the API Call
            if config:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config
                )
            else:
                 response = client.models.generate_content(
                    model=model_name,
                    contents=contents
                )
            
            # 2. If a schema is required, validate it immediately to catch json errors
            if response_schema:
                # If validation fails, it raises ValidationError, triggering the except block below
                return response_schema.model_validate_json(response.text)
            
            # 3. For non-schema calls (like images), just return the response
            return response

        except Exception as e:
            error_str = str(e)
            
            # Check for Rate Limit (429) or Quota Issues
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait_time = initial_delay * (2 ** attempt) + random.uniform(0, 1)
                st.warning(f"Rate limit hit ({model_name}). Retrying in {wait_time:.1f}s... (Attempt {attempt+1}/{retries}) error_str: {error_str}")
                time.sleep(wait_time)
                continue
            
            # Check for Pydantic Validation Errors (Malformed JSON)
            if response_schema and isinstance(e, ValidationError):
                st.warning(f"JSON Validation Error (Attempt {attempt+1}/{retries}). Retrying...")
                # Optional: Add a small delay even for validation errors
                time.sleep(2) 
                continue

            # For other known errors, you might want to stop or retry depending on severity
            st.error(f"API Error: {e}")
            return None
    
    st.error(f"Failed after {retries} retries.")
    return None


def crop_loose(image_paths, page_index, bboxes, q_id, output_dir, padding=70):
    """Performs the initial loose crop from the full page with heavy padding."""
    saved_files = []
    if not bboxes or page_index >= len(image_paths): return saved_files
    
    with Image.open(image_paths[page_index]) as img:
        w, h = img.size
        for i, bbox in enumerate(bboxes):
            ymin, xmin, ymax, xmax = bbox
            # Apply padding of 70 as requested
            left = max(0, (xmin - padding) * w / 1000)
            top = max(0, (ymin - padding) * h / 1000)
            right = min(w, (xmax + padding) * w / 1000)
            bottom = min(h, (ymax + padding) * h / 1000)
            
            fname = f"q{q_id}_loose_{i}.png"
            img.crop((left, top, right, bottom)).save(output_dir / fname)
            saved_files.append(fname)
    return saved_files

def generate_clean_diagram(client, loose_crop_path, output_path):
    """Uses gemini-2.5-flash-image to generate a brand new clean diagram image."""
    
    # Use the retry wrapper
    response = call_gemini_with_retry(
        client=client,
        model_name='gemini-2.5-flash-image',
        contents=[GENERATIVE_PROMPT, Image.open(loose_crop_path)]
    )

    if response:
        try:
            # Extract the image data from the response parts
            for part in response.parts:
                if part.inline_data:
                    part.as_image().save(output_path)
                    return True
        except Exception as e:
            st.warning(f"Error saving generated image: {e}")
            
    return False


# --- 4. Main App Logic ---

def main():
    st.title("📚 Professional MCQ & Diagram Generator")
    
    with st.sidebar:
        api_key = st.text_input("Gemini API Key", type="password")
        if not api_key: st.stop()
        client = genai.Client(api_key=api_key)

    uploaded_files = st.file_uploader("Upload Exam PDF", type="pdf", accept_multiple_files=True)

    if uploaded_files and st.button("🚀 Start Generative Extraction"):
        temp_dir = Path(tempfile.mkdtemp())
        img_dir, loose_dir, final_dir = temp_dir/"imgs", temp_dir/"loose", temp_dir/"final"
        for d in [img_dir, loose_dir, final_dir]: d.mkdir()

        # Step 1: PDF to High-Res Images
        pdf_paths = []
        for f in uploaded_files:
            p = temp_dir/f.name
            p.write_bytes(f.getbuffer())
            pdf_paths.append(p)
        
        matrix = fitz.Matrix(300/72, 300/72)
        all_pages = []
        for p in pdf_paths:
            doc = fitz.open(p)
            for i in range(len(doc)):
                out = img_dir / f"{p.stem}_p{i+1}.png"
                doc[i].get_pixmap(matrix=matrix, alpha=False).save(out)
                all_pages.append(out)

        # AI Loop
        all_q = []
        prog = st.progress(0, "Analyzing Content...")
        
        for idx, page_path in enumerate(all_pages):
            prog.progress((idx+1)/len(all_pages), f"Processing Page {idx+1}...")
            
            # Text & Loose Grounding
            # Text & Loose Grounding
            # USE RETRY FUNCTION HERE
            data = call_gemini_with_retry(
                client=client,
                model_name='gemini-2.5-pro',
                contents=[SYSTEM_INSTRUCTION, Image.open(page_path)],
                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=IntermediateResult),
                response_schema=IntermediateResult # Pass schema for validation inside the loop
            )
            
            if not data:
                st.error(f"Skipping page {idx+1} due to persistent errors.")
                continue

            # data is already a validated IntermediateResult object


            for q in data.questions:
                # 1. Physical Loose Crop (Padding 70)
                loose_fnames = crop_loose(all_pages, q.page_index, q.diagram_bboxes, q.id, loose_dir, padding=70)
                
                final_fnames = []
                for l_name in loose_fnames:
                    # 2. Generative Reproduction
                    clean_fname = f"clean_q{q.id}_{uuid.uuid4().hex[:4]}.png"
                    success = generate_clean_diagram(client, loose_dir/l_name, final_dir/clean_fname)
                    
                    if success:
                        final_fnames.append(clean_fname)
                    else:
                        # Fallback to loose crop if generation fails
                        shutil.copy(loose_dir/l_name, final_dir/l_name)
                        final_fnames.append(l_name)
                
                q.extracted_image_names = final_fnames
                all_q.append(q.model_dump())

        # Prepare ZIPs
        shutil.make_archive(str(temp_dir/"clean_diagrams"), 'zip', final_dir)
        shutil.make_archive(str(temp_dir/"loose_diagrams"), 'zip', loose_dir)
        
        st.success("✅ Extraction & Generation Complete!")
        
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Download Generated Diagrams (ZIP)", (temp_dir/"clean_diagrams.zip").read_bytes(), "clean_diagrams.zip")
        with c2:
            st.download_button("⬇️ Download Loose Crops (ZIP)", (temp_dir/"loose_diagrams.zip").read_bytes(), "loose_crops.zip")
        
        st.json(all_q)

if __name__ == "__main__":
    main()