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
import fitz # PyMuPDF


st.set_page_config(
    page_title="PDF MCQ Extractor (Gemini AI)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Pydantic Schemas (Updated for Stitching) ---

class QuestionData(BaseModel):
    id: int = Field(description="Literal printed question number.")
    chapter: str = Field(default="Physical World & Measurement", description="The chapter name, default or inferred.")
    question: str
    question_latex: str
    image_url: str = Field(default="", description="Descriptive tag if diagram is present.")
    options: List[str]
    difficulty: str = Field(description="Infer the difficulty: 'easy', 'medium', or 'hard'.")
    # NEW FIELD for stitching
    is_fragment: bool = Field(default=False, description="True if the question is incomplete and continues onto the next page.")
    is_fragment: bool = Field(default=False, description="True if the question is incomplete and continues onto the next page.")
    # NEW FIELDS for Image Extraction & Batching
    page_index: int = Field(default=0, description="The index (0-based) of the image in the provided batch where this question primarily appears.")
    diagram_bboxes: List[List[int]] = Field(default=[], description="Bounding boxes [ymin, xmin, ymax, xmax] for diagrams.")
    extracted_image_names: List[str] = Field(default=[], description="Filenames of extracted diagrams.")

class KeyData(BaseModel):
    id: int
    answer_option: str = Field(description="The single correct option letter (A, B, C, or D).")

class SolutionData(BaseModel):
    id: int
    solution: str = Field(description="The detailed solution, all math in $$. Use \\n for newlines.")
    # NEW FIELD for stitching
    is_fragment: bool = Field(default=False, description="True if the solution is incomplete and continues onto the next page.")

class IntermediateResult(BaseModel):
    """The ROOT schema used for every API call."""
    questions: List[QuestionData] = []
    answer_keys: List[KeyData] = []
    solutions: List[SolutionData] = []

# Final target schema for the output (Pass 2)
class FinalMCQ(BaseModel):
    id: int
    chapter: str
    question: str
    question_latex: str
    image_url: str
    options: List[str]
    answer: str
    solution: str
    difficulty: str
    marks: int = Field(default=4)
    extracted_image_names: List[str] = Field(default=[], description="List of filenames for extracted diagrams associated with this question.")

# --- 2. LLM System Instruction (Updated for Fragment Flagging) ---

SYSTEM_INSTRUCTION = """
You are an expert academic data extraction engine specializing in Physics, Chemistry, Mathematics, Biology exams. 
Your task is to analyze the provided single image of an exam page and accurately extract all distinct data fragments present: **Questions, Answer Keys, or Detailed Solutions.**

Your output MUST populate the corresponding lists (questions, answer_keys, or solutions) within the 'IntermediateResult' schema. If a section is absent from the image, return its list as an empty array ([]).

--- PRIMARY EXTRACTION RULES (ID Mapping) ---
1.  **Question/Solution ID Source:** The 'id' field MUST be the **literal printed number** (e.g., 1, 17, 30) visible in the image.
2.  **ID Continuity:** The literal ID must be used consistently across all three schemas.
3.  **Page Indexing:** For every Question, identify the `page_index` (0, 1, 2...) corresponding to the image in the input list where the question starts.

--- STRENGTHENED CONTENT EXTRACTION RULES ---
1.  **Fragment Flagging (CRITICAL):**
    * If a Question or Solution starts on this page but is clearly **truncated** or **incomplete** (e.g., the text abruptly stops mid-sentence, options are missing, or the solution steps are cut off at the bottom of the image), set the `is_fragment` field to `true`.
    * If the question/solution is complete on this page, set `is_fragment` to `false`.
    * **Note:** The external Python script will handle stitching based on this flag.
2.  **Questions & Options (Content):** Extract the **full body, options, and question_latex**.
3.  **Visual Grounding (Diagrams):** 
    * Identify EVERY diagram, graph, and technical illustration associated with a question.
    * For each one, provide a rectangular bounding box.
    * **Bounding Box Format:** Output as `[ymin, xmin, ymax, xmax]`.
    * **Scale:** Use normalized coordinates from 0 to 1000 where [0, 0] is top-left and [1000, 1000] is bottom-right.
4.  **Answer Keys (Arrangement):** Extract only the question ID and the final letter/option (A, B, C, D).
5.  **Solutions:** Extract the full step-by-step text for the `solution` field.

--- STRICT FORMATTING & QUALITY RULES ---
1.  **JSON Output:** The entire output MUST be a JSON object that strictly conforms to the 'IntermediateResult' Pydantic schema.
2.  **LaTeX Requirement:** All mathematical expressions, physics formulas, units (e.g., $\\text{kg}$), and dimensional analysis MUST be translated into standard **LaTeX format** and **strictly enclosed in single dollar signs ($$)**.
3.  **Solution Formatting:** In the `solution` field, replace native line breaks with the LaTeX newline command ('\\n').
4.  **Image URL:** Set the `image_url` to a **descriptive tag** (e.g., 'Image of bridge circuit diagram') if a diagram/graph is present. Otherwise, use an empty string ("").
5.  **Placeholders:** If an MCQ detail (like an option text or solution body) is unclear or missing, use an empty string (`""`) as a placeholder.
"""

def convert_pdfs_to_images(pdf_paths: List[Path], output_dir: Path, dpi: int = 300) -> List[Path]:
    """Converts PDF pages to PNG images and saves them to a temporary directory."""
    all_image_paths = []
    zoom_factor = dpi / 72
    matrix = fitz.Matrix(zoom_factor, zoom_factor)

    for pdf_path in pdf_paths:
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                
                # Use a cleaner filename structure for sorting later
                output_filename = f"{pdf_path.stem}_page_{page_num + 1}.png"
                output_path = output_dir / output_filename
                
                pix.save(output_path)
                all_image_paths.append(output_path)
            doc.close()
        except Exception as e:
            st.error(f"Error converting PDF {pdf_path.name}: {e}")
            continue
    return all_image_paths

def process_batch_with_ai(image_paths: List[Path], client: genai.Client, system_instruction: str, retries: int = 5, initial_delay: int = 10) -> Optional[IntermediateResult]:
    """
    Sends a BATCH of images to the Gemini model for structured extraction.
    """
    for attempt in range(retries):
        try:
            # Load all images in the batch
            images = [Image.open(p) for p in image_paths]
            
            # Construct content: System Instruction + List of Images
            content_payload = [system_instruction] + images
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=content_payload,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=IntermediateResult,
                )
            )
            # Use Pydantic to validate and parse the JSON string
            data = IntermediateResult.model_validate_json(response.text)
            return data
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait_time = initial_delay * (2 ** attempt) 
                st.warning(f"Rate limit hit for batch. Retrying in {wait_time} seconds... (Attempt {attempt+1}/{retries})")
                time.sleep(wait_time)
                continue
            elif isinstance(e, ValidationError):
                st.warning(f"Validation error for batch. Content may be corrupted. Error: {e}")
                return None
            else:
                st.error(f"AI API Error processing batch: {e}")
                return None
    
    st.error(f"Failed to process batch after {retries} retries.")
    return None
            


def crop_and_save_diagrams(image_paths: List[Path], page_index: int, bboxes: List[List[int]], q_id: int, output_dir: Path) -> List[str]:
    """Crops diagrams from the specific page in the batch based on bounding boxes."""
    saved_files = []
    if not bboxes:
        return saved_files
        
    # Validation: Ensure page_index is within bounds
    if page_index < 0 or page_index >= len(image_paths):
        # Fallback: Try cropping from the first image if index is invalid
        page_index = 0
        
    target_image_path = image_paths[page_index]

    try:
        with Image.open(target_image_path) as img:
            width, height = img.size
            for i, bbox in enumerate(bboxes):
                # bbox is [ymin, xmin, ymax, xmax] in normalized 0-1000 coords
                if len(bbox) != 4:
                    continue
                    
                ymin, xmin, ymax, xmax = bbox
                
                # Expand bounding box by 10 units (Over-fitting preference)
                padding = 30
                ymin = max(0, ymin - padding)
                xmin = max(0, xmin - padding)
                ymax = min(1000, ymax + padding)
                xmax = min(1000, xmax + padding)
                
                # Convert to pixel coordinates
                left = xmin * width / 1000
                top = ymin * height / 1000
                right = xmax * width / 1000
                bottom = ymax * height / 1000
                
                # Verify valid crop dimensions
                if right > left and bottom > top:
                    # Generate unique filename
                    filename = f"q{q_id}_diag_{i}_{uuid.uuid4().hex[:4]}.png"
                    filepath = output_dir / filename
                    
                    # Crop and save
                    img.crop((left, top, right, bottom)).save(filepath)
                    saved_files.append(filename)
    except Exception as e:
        st.warning(f"Failed to crop diagrams for question {q_id} from {target_image_path.name}: {e}")
        
    return saved_files

def natural_sort_key(s):
    """Sorts file names naturally (e.g., page_9 before page_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s.name)]

# Removed create_image_batches and stitch_fragments as they are no longer needed 
# for the native list-based batching approach. The model sees the whole context now.


def merge_data(all_q_data: List[QuestionData], all_key_data: List[KeyData], all_sol_data: List[SolutionData]) -> List[FinalMCQ]:
    """Consolidates data into the final structured output."""
    
    # With native list batching, the model sees the continuity, so explicit stitching is less critical,
    # but we can still group by ID just in case the model outputs fragments (it shouldn't if instructed well).
    # For now, we will treat the model instructions as robust enough to output complete questions.
    # If duplicates exist (e.g. from different batches), we take the first one.

    question_map: Dict[int, QuestionData] = {}
    key_map: Dict[int, str] = {k.id: k.answer_option for k in all_key_data}
    solution_map: Dict[int, str] = {s.id: s.solution for s in all_sol_data} 

    # Populate question map
    for q_data in all_q_data:
        if q_data.id not in question_map:
            question_map[q_data.id] = q_data
        else:
             # Basic Stitching Fallback: If ID exists, append text (unlikely with batching but safe)
             existing = question_map[q_data.id]
             existing.question += " " + q_data.question
             existing.options.extend(q_data.options)
             if q_data.question_latex:
                 existing.question_latex += " " + q_data.question_latex

    final_mcqs: List[FinalMCQ] = []
    sorted_ids = sorted(question_map.keys())

    # 3. Assign clean, sequential IDs and merge
    final_id_counter = 1
    for original_id in sorted_ids:
        q_data = question_map[original_id]
        
        # Look up Answer Key (A/B/C/D -> text)
        option_letter = key_map.get(original_id, "")
        answer_string = ""
        
        # Match the correct answer option letter to the actual option text
        if option_letter and q_data.options:
            option_index_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            index = option_index_map.get(option_letter.upper())
            
            if index is not None and 0 <= index < len(q_data.options):
                answer_string = q_data.options[index]
        
        # Determine the final image_url: Use the filename if an image was extracted
        final_image_url = q_data.image_url
        if q_data.extracted_image_names:
            # Use the first extracted image as the primary URL
            final_image_url = q_data.extracted_image_names[0]

        mcq = FinalMCQ(
            id=final_id_counter,
            chapter=q_data.chapter,
            question=q_data.question,
            question_latex=q_data.question_latex,
            image_url=final_image_url,
            options=q_data.options,
            answer=answer_string, 
            solution=solution_map.get(original_id, ""),
            difficulty=q_data.difficulty,
            marks=4,
            extracted_image_names=q_data.extracted_image_names
        )
        final_mcqs.append(mcq)
        final_id_counter += 1
        
    return final_mcqs

def main():
    st.title("📚 Academic MCQ Extractor")
    st.header("Upload PDF for Structured JSON Output")

    # --- Sidebar for Configuration ---
    with st.sidebar:
        st.header("API Key & Settings")
        
        # 1. Get API Key from User Input (Primary)
        user_api_key = st.text_input("Enter Gemini API Key", type="password", help="Get your key from https://aistudio.google.com/")
        
        # 2. Strict usage: Only use the user-provided key
        api_key = user_api_key

        if api_key:
            try:
                client = genai.Client(api_key=api_key)
                st.success("API Key configured successfully!")
            except Exception as e:
                st.error(f"Could not initialize Gemini Client: {e}")
                st.stop()
        else:
            st.warning("Please enter your Gemini API Key to proceed.")
            st.stop()

    # --- Main File Uploader ---
    # --- Initialize Session State ---
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = None
    if "json_result" not in st.session_state:
        st.session_state.json_result = None
    if "zip_bytes" not in st.session_state:
        st.session_state.zip_bytes = None
    if "preview_df" not in st.session_state:
        st.session_state.preview_df = None

    # --- Main File Uploader ---
    uploaded_files = st.file_uploader(
        "Upload your exam PDF files (Multiple files supported)",
        type="pdf",
        accept_multiple_files=True
    )

    start_button = st.button("🚀 Start Extraction Process")

    if uploaded_files and start_button:
        # Create a temporary directory structure for file management
        temp_dir = Path(tempfile.mkdtemp())
        pdf_storage_dir = temp_dir / "pdfs"
        image_output_dir = temp_dir / "processed_images"
        extracted_diagrams_dir = temp_dir / "extracted_diagrams"
        pdf_storage_dir.mkdir()
        image_output_dir.mkdir()
        extracted_diagrams_dir.mkdir()

        st.info(f"Starting pipeline using temporary directory: {temp_dir}")
        progress_bar = st.progress(0, text="Initializing...")
        
        try:
            # 1. Save uploaded files to the temporary directory
            pdf_paths_to_process = []
            for uploaded_file in uploaded_files:
                pdf_path = pdf_storage_dir / uploaded_file.name
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                pdf_paths_to_process.append(pdf_path)
            
            # 2. Phase 1: Convert PDFs to Images
            progress_bar.progress(10, text="Phase 1: Converting PDFs to Images...")
            generated_image_paths = convert_pdfs_to_images(
                pdf_paths_to_process, 
                image_output_dir
            )

            if not generated_image_paths:
                st.error("Could not generate any images from the uploaded PDFs. Please check the files.")
                return

            # Sort files naturally (fixes page_10 before page_2 issue)
            sorted_image_files = sorted(generated_image_paths, key=natural_sort_key)
            
            # --- OPTIMIZATION START: Batch Images ---
            # Batch size of 50 pages per request to minimize API calls globally.
            BATCH_SIZE = 50
            image_batches = [sorted_image_files[i:i + BATCH_SIZE] for i in range(0, len(sorted_image_files), BATCH_SIZE)]
            num_batches = len(image_batches)
            
            st.info(f"Optimization: Processing {len(sorted_image_files)} pages in {num_batches} batch(es) (Max {BATCH_SIZE} pages/batch).")
            # ----------------------------------------

            # 3. Phase 2: AI Extraction (Pass 1)
            all_q_data, all_key_data, all_sol_data = [], [], []

            st.subheader(f"🖼️ Processing {num_batches} Batches via Gemini AI...")
            
            status_text = st.empty()

            for i, batch_paths in enumerate(image_batches):
                progress_percent = 10 + int(80 * (i + 1) / num_batches)
                progress_bar.progress(progress_percent)
                status_text.text(f"Processing batch {i+1} of {num_batches}...")
                
                # Add a small delay to be polite to the API
                time.sleep(2) 
                
                result = process_batch_with_ai(batch_paths, client, SYSTEM_INSTRUCTION)
                
                if result:
                    # Crop diagrams immediately based on the current batch
                    for q in result.questions:
                        q.extracted_image_names = crop_and_save_diagrams(
                            batch_paths,
                            q.page_index,
                            q.diagram_bboxes, 
                            q.id, 
                            extracted_diagrams_dir
                        )

                    # Append all data extracted from this batch
                    all_q_data.extend(result.questions)
                    all_key_data.extend(result.answer_keys)
                    all_sol_data.extend(result.solutions)
            
            # 4. Phase 3: Merging & Finalization (Stitching and Final Merge)
            progress_bar.progress(95, text="Phase 3: Stitching data fragments and finalizing output...")
            final_mcqs = merge_data(all_q_data, all_key_data, all_sol_data)

            # --- Final Output Processing ---
            progress_bar.progress(100, text="Extraction Complete!")
            status_text.success(f"✅ Successfully extracted and merged {len(final_mcqs)} unique MCQs!")
            
            if final_mcqs:
                # Prepare JSON for download
                output_data_list = [mcq.model_dump() for mcq in final_mcqs]
                json_string = json.dumps(output_data_list, indent=2, ensure_ascii=False)
                st.session_state.json_result = json_string

                # Create ZIP of extracted images
                shutil.make_archive(str(temp_dir / "extracted_diagrams"), 'zip', extracted_diagrams_dir)
                zip_path = temp_dir / "extracted_diagrams.zip"
                
                if zip_path.exists():
                    with open(zip_path, "rb") as f:
                        st.session_state.zip_bytes = f.read()
                
                # Prepare Preview Data
                st.session_state.preview_df = [
                    {
                        "ID": m.id,
                        "Question Snippet": m.question[:80] + "...",
                        "Answer Found": "✅" if m.answer else "❌",
                        "Solution Found": "✅" if m.solution else "❌"
                    } for m in final_mcqs[:5]
                ]
                
                st.session_state.processed_data = True
            
        except Exception as e:
            st.error(f"An unexpected error occurred during the main pipeline execution: {e}")
            st.session_state.processed_data = False
            
        finally:
            # Crucial: Clean up the temporary directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                st.caption(f"Cleaned up temporary files.")

    # --- Display Results from Session State ---
    if st.session_state.processed_data:
        st.divider()
        st.header("Results")
        
        col1, col2 = st.columns(2)
        
        if st.session_state.json_result:
            with col1:
                st.download_button(
                    label="⬇️ Download Structured JSON File",
                    data=st.session_state.json_result,
                    file_name="mcq_extraction_results.json",
                    mime="application/json"
                )

        if st.session_state.zip_bytes:
            with col2:
                st.download_button(
                    label="⬇️ Download Extracted Images (ZIP)",
                    data=st.session_state.zip_bytes,
                    file_name="extracted_diagrams.zip",
                    mime="application/zip"
                )

        if st.session_state.preview_df:
            st.subheader("Extracted Questions Preview (First 5)")
            st.dataframe(st.session_state.preview_df)


if __name__ == "__main__":
    main()