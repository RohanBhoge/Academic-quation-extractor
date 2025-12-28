"""
Script utilizes OpenAI's GPT-4o Vision model to identify precise bounding boxes for technical diagrams within images. Handles image encoding and normalized coordinate parsing for accurate cropping.
"""
import os
import json
import re
import base64
import hashlib
from typing import Optional, Tuple
from PIL import Image, ImageDraw
import openai
# Set your key here - if it's empty, the script will now correctly error out
openai.api_key = os.getenv("OPENAI_API_KEY", "") 
IMAGE_PATH = "q3_diag_0_8be2.png" 
OUTPUT_IMAGE = "diagram_crop.jpg"
DEBUG_IMAGE = "debug_overlay.jpg"
MARGIN_PCT = 0.05 
def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
def get_bbox_vision(image_path: str) -> Optional[str]:
    """Uses GPT-4o to identify the diagram. This is much more accurate than CV."""
    img64 = encode_image(image_path)
    try:
        system_prompt="""Identify the bounding box for the complete technical diagram. 
INCLUDE: All geometric shapes (wedge B, block A), the ground surface texture, the angle '37°', and the 'x-y' coordinate axes. 
EXCLUDE: Any full sentences of question text above the diagram and any multiple-choice options (A, B, C, D) below the diagram. 
RETURN: ONLY a JSON list of [ymin, xmin, ymax, xmax] normalized to 0-1000."""
        client = openai.OpenAI(api_key=openai.api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text":system_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img64}"}}
                ]
            }],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Vision Model failed: {e}")
        return None
def crop_image(image_path: str, bbox_json: str):
    # Extract [ymin, xmin, ymax, xmax] from the string
    # Robustly find any 4 numbers in the string
    numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", bbox_json)
    if len(numbers) < 4:
        print(f"Could not parse coordinates from: {bbox_json}")
        return
    # Take the first 4 numbers
    ymin, xmin, ymax, xmax = map(float, numbers[:4])
    
    img = Image.open(image_path)
    w, h = img.size
    # Convert normalized to pixel coordinates
    left = (xmin / 1000) * w
    top = (ymin / 1000) * h
    right = (xmax / 1000) * w
    bottom = (ymax / 1000) * h
    # Add a small margin
    bw, bh = right - left, bottom - top
    left = max(0, left - bw * MARGIN_PCT)
    top = max(0, top - bh * MARGIN_PCT)
    right = min(w, right + bw * MARGIN_PCT)
    bottom = min(h, bottom + bh * MARGIN_PCT)
    crop = img.crop((left, top, right, bottom))
    crop.save(OUTPUT_IMAGE)
    
    # Save Debug
    dbg = img.copy()
    draw = ImageDraw.Draw(dbg)
    draw.rectangle([left, top, right, bottom], outline="red", width=5)
    dbg.save(DEBUG_IMAGE)
    print(f"✅ Accurate crop saved to {OUTPUT_IMAGE}")
if __name__ == "__main__":
    print("🔍 Requesting accurate coordinates from Vision Model...")
    bbox = get_bbox_vision(IMAGE_PATH)
    
    if bbox:
        print(f"Raw BBox from Model: {bbox}")
        crop_image(IMAGE_PATH, bbox)
    else:
        print("❌ Failed to get coordinates. Check your API Key.")