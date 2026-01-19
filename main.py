# This is a code for extracting questions, answer keys, and solutions from PDF files using AI. 

import streamlit as st
import time
import os
import json
import re
import shutil
import tempfile
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

# --- 1. Pydantic Schemas (Updated for Text-Only Extraction) ---

class QuestionData(BaseModel):
    id: int = Field(description="Literal printed question number.")
    chapter: str = Field(default="Physical World & Measurement", description="The chapter name, default or inferred.")
    question: str
    question_latex: str
    options: List[str]
    difficulty: str = Field(description="Infer the difficulty: 'Easy', 'Medium', or 'Hard'.")
    is_fragment: bool = Field(default=False, description="True if the question is incomplete and continues onto the next page.")

class KeyData(BaseModel):
    id: int
    answer_option: str = Field(description="The single correct option letter (A, B, C, or D).")

class SolutionData(BaseModel):
    id: int
    solution: str = Field(description="The detailed solution, all math in $$.")
    is_fragment: bool = Field(default=False, description="True if the solution is incomplete and continues onto the next page.")

class IntermediateResult(BaseModel):
    """The ROOT schema used for every API call."""
    questions: List[QuestionData] = []
    answer_keys: List[KeyData] = []
    solutions: List[SolutionData] = []

class FinalMCQ(BaseModel):
    id: int
    chapter: str
    question: str
    question_latex: str
    question_images: List[str] = Field(default=[])
    options: List[str]
    option_images: List[str] = Field(default=[])
    answer: str = Field(description="The correct answer option letter (A, B, C, or D).")
    solution: str
    solution_images: List[str] = Field(default=[])
    difficulty: str
    marks: int = Field(default=4)

# --- 2. LLM System Instruction (Updated for Text-Only Extraction) ---

SYSTEM_INSTRUCTION = """
You are an expert academic data extraction engine specializing in Physics, Chemistry, Mathematics, Biology exams. 
Your task is to analyze the provided images of exam pages and accurately extract all distinct data fragments present: **Questions, Answer Keys, or Detailed Solutions.**

**CRITICAL: TEXT-ONLY EXTRACTION**
You MUST extract ONLY text-based content and LaTeX expressions. **COMPLETELY IGNORE and SKIP any diagrams, tables, graphs, charts, or visual elements.** Do NOT extract questions that primarily rely on visual diagrams.

Your output MUST populate the corresponding lists (questions, answer_keys, or solutions) within the 'IntermediateResult' schema. If a section is absent from the image, return its list as an empty array ([]).

--- PRIMARY EXTRACTION RULES (ID Mapping) ---
1.  **Question/Solution ID Source:** The 'id' field MUST be the **literal printed number** (e.g., 1, 17, 30) visible in the image.
2.  **ID Continuity:** The literal ID must be used consistently across all three schemas.
3.  **Skip Visual Content:** If a question contains or requires a diagram, table, or chart to be answered, SKIP that question entirely.

--- CONTENT EXTRACTION RULES ---
1.  **Fragment Flagging:**
    * If a Question or Solution starts on this page but is clearly **truncated** or **incomplete** (e.g., the text abruptly stops mid-sentence, options are missing, or the solution steps are cut off), set the `is_fragment` field to `true`.
    * If the question/solution is complete on this page, set `is_fragment` to `false`.
2.  **Questions & Options (Content):** Extract the **full body, options, and question_latex** for text-based questions only.
3.  **Answer Keys:** Extract the correct answer letter (A, B, C, or D) as shown in the answer key section.

--- STRICT FORMATTING & QUALITY RULES ---
1.  **JSON Output:** The entire output MUST be a JSON object that strictly conforms to the 'IntermediateResult' Pydantic schema.
2.  **LaTeX Requirement:** All mathematical expressions, physics formulas, chemical equations, units (e.g., $\\text{kg}$), and symbols MUST be translated into standard **LaTeX format** and **strictly enclosed in single dollar signs ($)**.
3.  **Solution Formatting:** In the `solution` field, use proper formatting. Use newlines where appropriate.
4.  **Difficulty Levels:** Must be exactly one of: 'Easy', 'Medium', or 'Hard' (proper capitalization).
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
            

def natural_sort_key(s):
    """Sorts file names naturally (e.g., page_9 before page_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s.name)]

def merge_data(all_q_data: List[QuestionData], all_key_data: List[KeyData], all_sol_data: List[SolutionData], id_offset: int = 0) -> List[FinalMCQ]:
    """Consolidates data into the final structured output with sequential ID management."""

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

    # Assign sequential IDs starting from id_offset + 1
    final_id_counter = id_offset + 1
    for original_id in sorted_ids:
        q_data = question_map[original_id]
        
        # Get the answer letter (A/B/C/D) directly from key_map
        answer_letter = key_map.get(original_id, "")

        mcq = FinalMCQ(
            id=final_id_counter,
            chapter=q_data.chapter,
            question=q_data.question,
            question_latex=q_data.question_latex,
            question_images=[],  # Empty as per requirements
            options=q_data.options,
            option_images=[],  # Empty as per requirements
            answer=answer_letter,  # Just the letter (A/B/C/D)
            solution=solution_map.get(original_id, ""),
            solution_images=[],  # Empty as per requirements
            difficulty=q_data.difficulty,
            marks=4
        )
        final_mcqs.append(mcq)
        final_id_counter += 1
        
    return final_mcqs

def main():
    st.title("📚 Academic MCQ Extractor (Queue-Based)")
    st.header("Upload PDFs for Structured JSON Output")

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

    # --- Initialize Session State for Queue Management ---
    if "all_mcqs" not in st.session_state:
        st.session_state.all_mcqs = []
    if "question_id_offset" not in st.session_state:
        st.session_state.question_id_offset = 0
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = False
    if "pdf_queue" not in st.session_state:
        st.session_state.pdf_queue = []

    # Display current stats
    st.info(f"📊 Total Questions Extracted: {len(st.session_state.all_mcqs)} | Next Question ID will start from: {st.session_state.question_id_offset + 1}")

    # --- Main File Uploader ---
    uploaded_files = st.file_uploader(
        "Upload your exam PDF files (Multiple files supported - processed in order)",
        type="pdf",
        accept_multiple_files=True
    )

    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("🚀 Start Extraction Process")
    with col2:
        reset_button = st.button("🔄 Reset All Data")

    if reset_button:
        st.session_state.all_mcqs = []
        st.session_state.question_id_offset = 0
        st.session_state.processed_data = False
        st.session_state.pdf_queue = []
        st.success("✅ All data has been reset!")
        st.rerun()

    if uploaded_files and start_button:
        # Create a temporary directory structure for file management
        temp_dir = Path(tempfile.mkdtemp())
        pdf_storage_dir = temp_dir / "pdfs"
        image_output_dir = temp_dir / "processed_images"
        pdf_storage_dir.mkdir()
        image_output_dir.mkdir()

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
            
            st.success(f"📁 Processing {len(pdf_paths_to_process)} PDF(s) in queue order...")
            
            # Process each PDF in queue order
            for pdf_idx, pdf_path in enumerate(pdf_paths_to_process):
                st.subheader(f"🔄 Processing PDF {pdf_idx + 1}/{len(pdf_paths_to_process)}: {pdf_path.name}")
                
                # 2. Phase 1: Convert PDF to Images
                progress_bar.progress(10 + (pdf_idx * 80 // len(pdf_paths_to_process)), 
                    text=f"Phase 1: Converting PDF {pdf_idx + 1} to Images...")
                
                generated_image_paths = convert_pdfs_to_images(
                    [pdf_path],  # Process one PDF at a time
                    image_output_dir
                )

                if not generated_image_paths:
                    st.error(f"Could not generate images from {pdf_path.name}. Skipping...")
                    continue

                # Sort files naturally (fixes page_10 before page_2 issue)
                sorted_image_files = sorted(generated_image_paths, key=natural_sort_key)
                
                # --- Batch Images ---
                BATCH_SIZE = 50
                image_batches = [sorted_image_files[i:i + BATCH_SIZE] for i in range(0, len(sorted_image_files), BATCH_SIZE)]
                num_batches = len(image_batches)
                
                st.info(f"  📄 {len(sorted_image_files)} pages in {num_batches} batch(es)")
                
                # 3. Phase 2: AI Extraction
                all_q_data, all_key_data, all_sol_data = [], [], []
                status_text = st.empty()

                for i, batch_paths in enumerate(image_batches):
                    batch_progress = 10 + (pdf_idx * 80 // len(pdf_paths_to_process)) + (70 * (i + 1) // (num_batches * len(pdf_paths_to_process)))
                    progress_bar.progress(batch_progress)
                    status_text.text(f"  Processing batch {i+1}/{num_batches} of PDF {pdf_idx+1}...")
                    
                    # Add a small delay to be polite to the API
                    time.sleep(2) 
                    
                    result = process_batch_with_ai(batch_paths, client, SYSTEM_INSTRUCTION)
                    
                    if result:
                        # Append all data extracted from this batch
                        all_q_data.extend(result.questions)
                        all_key_data.extend(result.answer_keys)
                        all_sol_data.extend(result.solutions)
                
                # 4. Phase 3: Merge with sequential ID management
                progress_bar.progress(85, text=f"Phase 3: Merging data for PDF {pdf_idx + 1}...")
                
                # Use the current offset for this PDF's questions
                current_offset = st.session_state.question_id_offset
                pdf_mcqs = merge_data(all_q_data, all_key_data, all_sol_data, id_offset=current_offset)
                
                # Update offset for next PDF
                if pdf_mcqs:
                    st.session_state.question_id_offset += len(pdf_mcqs)
                    st.session_state.all_mcqs.extend(pdf_mcqs)
                    status_text.success(f"  ✅ Extracted {len(pdf_mcqs)} questions from {pdf_path.name} (IDs: {pdf_mcqs[0].id} to {pdf_mcqs[-1].id})")

            progress_bar.progress(100, text="Extraction Complete!")
            st.success(f"🎉 Successfully extracted {len(st.session_state.all_mcqs)} total questions across all PDFs!")
            st.session_state.processed_data = True
            
        except Exception as e:
            st.error(f"An unexpected error occurred during the main pipeline execution: {e}")
            st.session_state.processed_data = False
            
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                st.caption(f"Cleaned up temporary files.")

    if st.session_state.processed_data and st.session_state.all_mcqs:
        st.divider()
        st.header("Results")
        
        # Generate JSON from all accumulated MCQs
        output_data_list = [mcq.model_dump() for mcq in st.session_state.all_mcqs]
        json_string = json.dumps(output_data_list, indent=2, ensure_ascii=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="⬇️ Download Complete JSON File",
                data=json_string,
                file_name="mcq_extraction_results.json",
                mime="application/json"
            )
        
        with col2:
            st.metric("Total Questions", len(st.session_state.all_mcqs))

        # Preview first 10 questions
        st.subheader("Extracted Questions Preview (First 10)")
        preview_data = [
            {
                "ID": m.id,
                "Chapter": m.chapter,
                "Question Snippet": m.question[:60] + "...",
                "Answer": m.answer,
                "Difficulty": m.difficulty
            } for m in st.session_state.all_mcqs[:10]
        ]
        st.dataframe(preview_data, use_container_width=True)


if __name__ == "__main__":
    main()