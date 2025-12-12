import streamlit as st
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

# --- 2. LLM System Instruction (Updated for Fragment Flagging) ---

SYSTEM_INSTRUCTION = """
You are an expert academic data extraction engine specializing in Physics, Chemistry, Mathematics, Biology exams. 
Your task is to analyze the provided single image of an exam page and accurately extract all distinct data fragments present: **Questions, Answer Keys, or Detailed Solutions.**

Your output MUST populate the corresponding lists (questions, answer_keys, or solutions) within the 'IntermediateResult' schema. If a section is absent from the image, return its list as an empty array ([]).

--- PRIMARY EXTRACTION RULES (ID Mapping) ---
1.  **Question/Solution ID Source:** The 'id' field MUST be the **literal printed number** (e.g., 1, 17, 30) visible in the image.
2.  **ID Continuity:** The literal ID must be used consistently across all three schemas.

--- STRENGTHENED CONTENT EXTRACTION RULES ---
1.  **Fragment Flagging (CRITICAL):**
    * If a Question or Solution starts on this page but is clearly **truncated** or **incomplete** (e.g., the text abruptly stops mid-sentence, options are missing, or the solution steps are cut off at the bottom of the image), set the `is_fragment` field to `true`.
    * If the question/solution is complete on this page, set `is_fragment` to `false`.
    * **Note:** The external Python script will handle stitching based on this flag.
2.  **Questions & Options (Content):** Extract the **full body, options, and question\_latex**.
3.  **Answer Keys (Arrangement):** Extract only the question ID and the final letter/option (A, B, C, D).
4.  **Solutions:** Extract the full step-by-step text for the `solution` field.

--- STRICT FORMATTING & QUALITY RULES ---
1.  **JSON Output:** The entire output MUST be a JSON object that strictly conforms to the 'IntermediateResult' Pydantic schema.
2.  **LaTeX Requirement:** All mathematical expressions, physics formulas, units (e.g., $\\text{kg}$), and dimensional analysis MUST be translated into standard **LaTeX format** and **strictly enclosed in single dollar signs ($$)**.
3.  **Solution Formatting:** In the `solution` field, replace native line breaks with the LaTeX newline command ('\\n').
4.  **Image URL:** Set the `image_url` to a **descriptive tag** (e.g., 'Image of bridge circuit diagram') if a diagram/graph is present. Otherwise, use an empty string ("").
5.  **Placeholders:** If an MCQ detail (like an option text or solution body) is unclear or missing, use an empty string (`""`) as a placeholder.
"""

def convert_pdfs_to_images(pdf_paths: List[Path], output_dir: Path, dpi: int = 300) -> List[Path]:
    """Converts PDF pages to PNG images and saves them to a temporary directory."""
    # (Function remains the same as it correctly handles PDF to page images)
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

def process_image_with_ai(image_path: Path, client: genai.Client, system_instruction: str) -> Optional[IntermediateResult]:
    """Sends a single image to the Gemini model for structured extraction."""
    try:
        img = Image.open(image_path)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[system_instruction, img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=IntermediateResult,
            )
        )
        # Use Pydantic to validate and parse the JSON string
        data = IntermediateResult.model_validate_json(response.text)
        return data
    except ValidationError as e:
        st.warning(f"Validation error for {image_path.name}. Content may be corrupted. Error: {e}")
        return None
    except Exception as e:
        st.error(f"AI API Error processing {image_path.name}: {e}")
        return None

def natural_sort_key(s):
    """Sorts file names naturally (e.g., page_9 before page_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s.name)]

# NEW FUNCTION: Contextual Stitching
def stitch_fragments(data_list: List[Union[QuestionData, SolutionData]]) -> List[Union[QuestionData, SolutionData]]:
    """
    Stitches together fragmented data (questions or solutions) based on ID and
    the implicit page order of the input list.
    """
    
    # 1. Group all fragments by their original ID
    grouped_data: Dict[int, List[Union[QuestionData, SolutionData]]] = {}
    for item in data_list:
        grouped_data.setdefault(item.id, []).append(item)

    stitched_list: List[Union[QuestionData, SolutionData]] = []
    
    for item_id, fragments in grouped_data.items():
        # The list 'fragments' is already in page order due to the input processing flow.
        
        # Take the first instance as the base item
        primary_item = fragments[0]
        
        # Check if the list contains QuestionData or SolutionData
        is_question = isinstance(primary_item, QuestionData)

        # Iterate over the remaining fragments, starting from the second one
        for subsequent_item in fragments[1:]:
            
            if is_question:
                # Stitch Question and Options
                primary_item.question += " " + subsequent_item.question
                primary_item.options.extend(subsequent_item.options)
                
                # Merge LaTeX fields only if the subsequent one is non-empty
                if subsequent_item.question_latex and subsequent_item.question_latex != primary_item.question_latex:
                     primary_item.question_latex += " \\text{ [Cont.] } " + subsequent_item.question_latex
                
                # Merge image URLs if the subsequent one is not empty (e.g., diagram on the next page)
                if subsequent_item.image_url and subsequent_item.image_url != primary_item.image_url:
                    primary_item.image_url += " | " + subsequent_item.image_url
            
            else: # Must be SolutionData
                # Stitch Solution text
                primary_item.solution += subsequent_item.solution
        
        # Set the final fragment status to False after merging
        primary_item.is_fragment = False 
        
        stitched_list.append(primary_item)

    return stitched_list


def merge_data(all_q_data: List[QuestionData], all_key_data: List[KeyData], all_sol_data: List[SolutionData]) -> List[FinalMCQ]:
    """Consolidates and stitches fragmented data into the final structured output."""
    
    # 1. STITCH FRAGMENTS FIRST
    # We must cast the lists for type hinting in stitch_fragments
    stitched_q_data = stitch_fragments(all_q_data)
    stitched_sol_data = stitch_fragments(all_sol_data)

    question_map: Dict[int, QuestionData] = {}
    key_map: Dict[int, str] = {k.id: k.answer_option for k in all_key_data}
    solution_map: Dict[int, str] = {s.id: s.solution for s in stitched_sol_data} # Use stitched solutions

    # 2. Populate the master question map (using stitched questions)
    for q_data in stitched_q_data:
        # If multiple fragments were found, this ensures only the fully stitched object is stored
        if q_data.id not in question_map:
            question_map[q_data.id] = q_data

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
            
        # Build the final structured MCQ object
        mcq = FinalMCQ(
            id=final_id_counter,
            chapter=q_data.chapter,
            question=q_data.question,
            question_latex=q_data.question_latex,
            image_url=q_data.image_url,
            options=q_data.options,
            answer=answer_string, 
            solution=solution_map.get(original_id, ""),
            difficulty=q_data.difficulty,
            marks=4
        )
        final_mcqs.append(mcq)
        final_id_counter += 1
        
    return final_mcqs


# --- 4. Streamlit UI and Execution Logic (main) ---

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
    uploaded_files = st.file_uploader(
        "Upload your exam PDF files (Multiple files supported)",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files and st.button("🚀 Start Extraction Process"):
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
            num_images = len(sorted_image_files)

            # 3. Phase 2: AI Extraction (Pass 1)
            all_q_data, all_key_data, all_sol_data = [], [], []

            st.subheader(f"🖼️ Processing {num_images} Images via Gemini AI...")
            
            status_text = st.empty()

            for i, image_path in enumerate(sorted_image_files):
                progress_percent = 10 + int(80 * (i + 1) / num_images)
                progress_bar.progress(progress_percent)
                status_text.text(f"Processing image {i+1} of {num_images}: {image_path.name}")
                
                result = process_image_with_ai(image_path, client, SYSTEM_INSTRUCTION)
                
                if result:
                    # Append all data extracted from this page
                    all_q_data.extend(result.questions)
                    all_key_data.extend(result.answer_keys)
                    all_sol_data.extend(result.solutions)
            
            # 4. Phase 3: Merging & Finalization (Stitching and Final Merge)
            progress_bar.progress(95, text="Phase 3: Stitching data fragments and finalizing output...")
            final_mcqs = merge_data(all_q_data, all_key_data, all_sol_data)

            # --- Final Output ---
            progress_bar.progress(100, text="Extraction Complete!")
            status_text.success(f"✅ Successfully extracted and merged {len(final_mcqs)} unique MCQs!")
            
            if final_mcqs:
                # Prepare JSON for download
                output_data_list = [mcq.model_dump() for mcq in final_mcqs]
                json_string = json.dumps(output_data_list, indent=2, ensure_ascii=False)
                
                st.subheader("Download Results")
                st.download_button(
                    label="⬇️ Download Structured JSON File",
                    data=json_string,
                    file_name="mcq_extraction_results.json",
                    mime="application/json"
                )

                st.subheader("Extracted Questions Preview (First 5)")
                
                preview_data = [
                    {
                        "ID": m.id,
                        "Question Snippet": m.question[:80] + "...",
                        "Answer Found": "✅" if m.answer else "❌",
                        "Solution Found": "✅" if m.solution else "❌"
                    } for m in final_mcqs[:5]
                ]
                st.dataframe(preview_data)

        except Exception as e:
            st.error(f"An unexpected error occurred during the main pipeline execution: {e}")
            
        finally:
            # Crucial: Clean up the temporary directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                st.caption(f"Cleaned up temporary files.")


if __name__ == "__main__":
    main()