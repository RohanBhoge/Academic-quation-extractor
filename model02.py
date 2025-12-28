"""
Standalone script for precise diagram bounding box detection. Implements a hybrid approach using Computer Vision (OpenCV) for deterministic cropping with a fallback to Gemini 1.5 Pro/Flash for semantic understanding. Features result caching.
"""
import os
import json
import re
import hashlib
from typing import Optional, Tuple

import google.generativeai as genai
from PIL import Image, ImageDraw

# Optional: deterministic CV-based cropper (fallback)
try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

# ---------------- CONFIGURATION ---------------- #
# Prefer environment variable; falls back to the value below if set.
API_KEY = os.getenv("GEMINI_API_KEY", "")

# Path to your image file (we'll auto-detect if missing)
IMAGE_PATH = r"C:\Users\bhoge\OneDrive\Documents\Desktop\PDF_Latex_Extraction\q3_diag_0_8be2.png"

# Extra padding around detected box (as fraction of box size)
MARGIN_PCT = 0.08  # 8%

# Optional upscale factor for the final crop (1.0 = no upscale)
UPSCALE = 1.5

# Save an overlay image showing the crop box on the original
SAVE_DEBUG_OVERLAY = True
# ----------------------------------------------- #

# Lightweight cache so identical images reuse coordinates (fewer Gemini calls)
_CACHE_FILE = "crop_cache.json"

def _img_hash(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(8192)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _load_cache() -> dict:
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_cache(cache: dict) -> None:
    try:
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass

def _find_image_fallback(path: str) -> str:
    """Return a usable image path. If the provided path doesn't exist,
    try likely filenames and finally the first image found in the folder."""
    if os.path.exists(path):
        return path
    candidates = [
        "img.jpg", "img.jpeg", "image.jpg", "image.jpeg", "image.png", "img.png",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    for f in os.listdir("."):
        if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}:
            return f
    return path


def get_accurate_crop_coordinates(image_path, api_key):
    """
    Sends the image to Gemini 1.5 Pro to identify the exact bounding box
    of the diagram, excluding surrounding question text.
    """
    genai.configure(api_key=api_key)
    
    # Use gemini-2.0-flash-exp for best spatial reasoning with determinism
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    img = Image.open(image_path)

    # This prompt is the "Brain" of the operation. It instructs Gemini exactly what to keep.
    prompt = """
    Analyze this image and identify the bounding box for the physics diagram area.
    
    CRITICAL RULES:
    1. INCLUDE: The graphical illustration (piston/container/T-bar) AND labels directly attached to the diagram (e.g., 'h1=1.5 m', 'u1', 'a=2.66 m/s^2', 'h2').
    2. EXCLUDE: Question/answer text above and paragraphs below that are not part of the diagram.
    3. Ensure the box is slightly generous so no label/bracket is clipped.
    
    Output ONLY a JSON list representing the bounding box in the format [ymin, xmin, ymax, xmax].
    The coordinates must be on a scale of 0 to 1000 (normalized).
    Example: [150, 200, 850, 800]
    """

    try:
        response = model.generate_content(
            [prompt, img],
            generation_config={
                "temperature": 0,
                "top_p": 1,
                "candidate_count": 1,
                "response_mime_type": "application/json",
            },
        )
        return response.text
    except Exception as e:
        print(f"Error communicating with Gemini: {e}")
        return None

def _parse_box(box_response: str) -> Optional[Tuple[float, float, float, float]]:
    """Extract [ymin, xmin, ymax, xmax] from model/CV text; accept nested lists and union."""
    if not box_response:
        return None
    try:
        clean_text = re.search(r"\[.*\]", box_response, re.DOTALL).group(0)
        data = json.loads(clean_text)
        # [[...]] or multiple boxes -> union into one
        if isinstance(data, list) and data and isinstance(data[0], list):
            boxes = [b for b in data if isinstance(b, list) and len(b) == 4]
            ymin = min(b[0] for b in boxes)
            xmin = min(b[1] for b in boxes)
            ymax = max(b[2] for b in boxes)
            xmax = max(b[3] for b in boxes)
            return float(ymin), float(xmin), float(ymax), float(xmax)
        if isinstance(data, list) and len(data) == 4:
            ymin, xmin, ymax, xmax = data
            return float(ymin), float(xmin), float(ymax), float(xmax)
        return None
    except Exception:
        return None


def _cv_detect_box(image_path: str) -> Optional[str]:
    """Deterministic crop via color/edge heuristics. Returns JSON list string.
    Prefers purple-stroke diagrams; falls back to edge-based if color mask is weak."""
    if cv2 is None or np is None:
        return None
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Purple mask in HSV (tune as needed for your source images)
    lower = np.array([125, 30, 30])
    upper = np.array([165, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    ys, xs = np.where(mask > 0)
    if ys.size < 500:  # fallback to edge-based if little purple is found
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 180)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        nz = cv2.findNonZero(edges)
        if nz is None:
            return None
        xs = nz[:, 0, 0]
        ys = nz[:, 0, 1]

    xmin = max(0, int(xs.min()))
    xmax = min(w, int(xs.max()))
    ymin = max(0, int(ys.min()))
    ymax = min(h, int(ys.max()))

    # Normalize to 0..1000 scale to reuse the same cropper
    norm = [ymin * 1000.0 / h, xmin * 1000.0 / w, ymax * 1000.0 / h, xmax * 1000.0 / w]
    return json.dumps([round(norm[0], 2), round(norm[1], 2), round(norm[2], 2), round(norm[3], 2)])


def _purple_bbox(image_path: str) -> Optional[Tuple[float, float, float, float]]:
    """Compute a deterministic bounding box that tightly encloses purple strokes/text.
    Returns normalized [ymin, xmin, ymax, xmax] or None if OpenCV unavailable or mask empty."""
    if cv2 is None or np is None:
        return None
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Purple mask; slightly wider range than CV fallback to catch faint text
    lower = np.array([120, 20, 20])
    upper = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None

    xmin = max(0, int(xs.min()))
    xmax = min(w, int(xs.max()))
    ymin = max(0, int(ys.min()))
    ymax = min(h, int(ys.max()))

    return (
        round(ymin * 1000.0 / h, 2),
        round(xmin * 1000.0 / w, 2),
        round(ymax * 1000.0 / h, 2),
        round(xmax * 1000.0 / w, 2),
    )

def _refine_box_with_purple(image_path: str, coords: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Expand the box to include nearby purple strokes/text so labels like 'h2' aren't clipped.
    Works deterministically and avoids extra Gemini calls."""
    if cv2 is None or np is None:
        return coords
    ymin, xmin, ymax, xmax = coords
    img = cv2.imread(image_path)
    if img is None:
        return coords
    h, w = img.shape[:2]

    # Convert box to pixel coords
    left = int((xmin / 1000.0) * w)
    top = int((ymin / 1000.0) * h)
    right = int((xmax / 1000.0) * w)
    bottom = int((ymax / 1000.0) * h)

    # Purple mask over the whole image
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([125, 30, 30])
    upper = np.array([165, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Consider a search region expanded around current box to pick nearby labels
    pad_x = int(0.10 * (right - left) + 16)
    pad_y = int(0.10 * (bottom - top) + 16)
    sx = max(0, left - pad_x)
    sy = max(0, top - pad_y)
    ex = min(w, right + pad_x)
    ey = min(h, bottom + pad_y)

    region = mask[sy:ey, sx:ex]
    ys, xs = np.where(region > 0)
    if ys.size > 0:
        # Purple extent within region -> map back to image coords
        xmin_p = sx + int(xs.min())
        xmax_p = sx + int(xs.max())
        ymin_p = sy + int(ys.min())
        ymax_p = sy + int(ys.max())
        # Expand box to include purple extent
        left = min(left, xmin_p)
        top = min(top, ymin_p)
        right = max(right, xmax_p)
        bottom = max(bottom, ymax_p)

    # Convert back to normalized coordinates
    ymin_n = round(top * 1000.0 / h, 2)
    xmin_n = round(left * 1000.0 / w, 2)
    ymax_n = round(bottom * 1000.0 / h, 2)
    xmax_n = round(right * 1000.0 / w, 2)
    return (ymin_n, xmin_n, ymax_n, xmax_n)


def crop_image(image_path, box_response):
    """
    Parses the JSON response and crops the image using Pillow.
    """
    try:
        coords = _parse_box(box_response)
        # Prefer deterministic purple-only bounding box for logical diagram cropping
        purple = _purple_bbox(image_path)
        if purple is not None:
            ymin, xmin, ymax, xmax = purple
        else:
            if not coords:
                raise ValueError("Could not parse bounding box from model response and no CV available")
            coords = _refine_box_with_purple(image_path, coords)
            ymin, xmin, ymax, xmax = coords

        img = Image.open(image_path)
        width, height = img.size

        # Convert normalized 0-1000 coordinates to actual pixels
        left = (xmin / 1000.0) * width
        top = (ymin / 1000.0) * height
        right = (xmax / 1000.0) * width
        bottom = (ymax / 1000.0) * height

        # Expand by a percentage-based margin so labels/brackets aren't clipped
        box_w = max(1.0, right - left)
        box_h = max(1.0, bottom - top)
        mx = box_w * MARGIN_PCT
        my = box_h * MARGIN_PCT
        left = max(0, left - mx)
        top = max(0, top - my)
        right = min(width, right + mx)
        bottom = min(height, bottom + my)

        # Perform the crop
        cropped_img = img.crop((left, top, right, bottom))
        if UPSCALE and UPSCALE > 1.0:
            new_w = int(cropped_img.width * UPSCALE)
            new_h = int(cropped_img.height * UPSCALE)
            cropped_img = cropped_img.resize((new_w, new_h), Image.LANCZOS)
        
        # Save the result
        output_filename = "refined_diagram_crop.jpg"
        cropped_img.save(output_filename)
        print(f"✅ Success! Cropped image saved as '{output_filename}'")
        print(f"   Coordinates found (normalized): {[ymin, xmin, ymax, xmax]}")

        # Optional debug overlay of crop rectangle
        if SAVE_DEBUG_OVERLAY:
            overlay = img.copy()
            draw = ImageDraw.Draw(overlay)
            draw.rectangle([left, top, right, bottom], outline=(255, 0, 0), width=4)
            overlay_path = "debug_box_overlay.jpg"
            overlay.save(overlay_path)
            print(f"   Debug overlay saved as '{overlay_path}'")
        
        # Show the image (opens in your default viewer)
        cropped_img.show()

    except Exception as e:
        print(f"Error processing the crop: {e}")
        print(f"Raw API Response: {box_response}")

# ---------------- EXECUTION ---------------- #
if __name__ == "__main__":
    img_path = _find_image_fallback(IMAGE_PATH)
    if not os.path.exists(img_path):
        print(f"❌ Could not find an image at '{IMAGE_PATH}'. Place an image here or update IMAGE_PATH.")
        raise SystemExit(1)

    cache = _load_cache()
    ih = _img_hash(img_path)
    coordinates_json = cache.get(ih)

    if not coordinates_json:
        # First try deterministic CV crop; if not satisfactory, fall back to Gemini once
        coordinates_json = _cv_detect_box(img_path)
        if not coordinates_json:
            print("CV fallback failed; querying Gemini...")
            print(f"Analyzing '{img_path}' with Gemini Vision...")
            coordinates_json = get_accurate_crop_coordinates(img_path, API_KEY)
        if coordinates_json:
            cache[ih] = coordinates_json
            _save_cache(cache)

    if coordinates_json:
        crop_image(img_path, coordinates_json)
        print("All done.")  