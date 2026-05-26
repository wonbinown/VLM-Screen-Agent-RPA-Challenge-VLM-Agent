import base64
import io
import json
import re
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageOps


# =========================================================
# Streamlit 기본 설정
# =========================================================
st.set_page_config(
    page_title="RPA Challenge VLM Agent",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# 상수
# =========================================================
TARGET_FIELDS = [
    "First Name",
    "Last Name",
    "Company Name",
    "Role in Company",
    "Address",
    "Email",
    "Phone Number",
]


# =========================================================
# GPT-4o Vision용 System Prompt
# =========================================================
SYSTEM_PROMPT = """
You are a Vision-Language AI Agent specialized in RPA Challenge UI automation.

Your job is NOT to estimate raw x/y coordinates directly.
Your job is to match each target field label to one of the pre-detected input-line candidates.

The input-line candidates are detected by computer vision and are marked on the image with blue boxes and labels such as C1, C2, C3.

Target fields:
1. First Name
2. Last Name
3. Company Name
4. Role in Company
5. Address
6. Email
7. Phone Number

RPA Challenge UI facts:
- RPA Challenge input fields usually look like thin gray underline fields, not normal rectangular boxes.
- Field positions randomly change after each submission.
- Fields may appear in two or three columns.
- Do not infer positions from a fixed order.
- Match each text label to the nearest visually corresponding underline input field.
- You must choose only from the provided candidate IDs.

Critical validation rules:
- Each candidate_id can be assigned to at most one field.
- Never reuse the same candidate_id for multiple fields.
- If you are not sure, set found=false and candidate_id=null.
- Do not invent candidate IDs.
- Do not return x/y coordinates.
- Return all 7 fields in the exact target order.
- The program will validate your output. Duplicate or invalid candidate_id will be rejected.

Return ONLY a valid JSON array.
Do not include Markdown.
Do not include ```json.
Do not include explanation outside JSON.
Do not include comments.

Each array item must follow exactly this structure:
{
  "field_name": "First Name",
  "found": true,
  "candidate_id": 1,
  "confidence": 0.0,
  "visual_evidence": "The First Name label is closest to candidate C1.",
  "mouse_action_plan": [
    {
      "step": 1,
      "action": "Move mouse to the center of the selected candidate input line."
    },
    {
      "step": 2,
      "action": "Click the input line."
    },
    {
      "step": 3,
      "action": "Type the corresponding value for this field."
    }
  ]
}
"""


def build_user_prompt(candidates: list) -> str:
    candidate_lines = []
    for c in candidates:
        candidate_lines.append(
            {
                "candidate_id": c["candidate_id"],
                "center_percent": c["center"],
                "bounding_box_percent": c["bounding_box"],
                "source_methods": c.get("source_methods", []),
            }
        )

    return f"""
This image shows an RPA Challenge form.

Computer vision has already detected possible thin underline input fields.
Each candidate is marked on the image with a blue box and a candidate label such as C1, C2, C3.

Candidate list:
{json.dumps(candidate_lines, ensure_ascii=False, indent=2)}

Your task:
Match each of these 7 target fields to the correct candidate_id:
- First Name
- Last Name
- Company Name
- Role in Company
- Address
- Email
- Phone Number

Important:
- Do not estimate coordinates.
- Select only from the provided candidate IDs.
- Do not reuse the same candidate_id for multiple fields.
- Each field must map to a unique input-line candidate.
- Match the field label to the nearest visually corresponding underline input field.
- RPA Challenge fields can appear in random positions.
- Do not assume a fixed order.
- If a field cannot be confidently matched to a unique candidate, set found=false and candidate_id=null.

Return ONLY a valid JSON array.
"""


# =========================================================
# 이미지 인코딩
# =========================================================
def image_to_base64_png(image: Image.Image, max_size: int = 1600) -> str:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    max_dim = max(width, height)

    if max_dim > max_size:
        scale = max_size / max_dim
        image = image.resize((int(width * scale), int(height * scale)))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# =========================================================
# JSON 파싱
# =========================================================
def extract_json_array(text: str):
    if not text:
        raise ValueError("모델 응답이 비어 있습니다.")

    cleaned = text.strip()
    cleaned = (
        cleaned.replace("```json", "")
        .replace("```JSON", "")
        .replace("```", "")
        .strip()
    )

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "fields" in parsed and isinstance(parsed["fields"], list):
            return parsed["fields"]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[\s*{.*}\s*\]", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("JSON 배열을 파싱하지 못했습니다. 원본 응답을 확인하세요.")


def clamp_percent(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(100.0, value))


def clamp_confidence(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def parse_candidate_id(value):
    if value is None:
        return None

    if isinstance(value, int):
        return value

    text = str(value).strip()
    match = re.search(r"\d+", text)
    if match:
        return int(match.group(0))

    return None


# =========================================================
# Bounding Box 유틸
# =========================================================
def box_iou(a: dict, b: dict) -> float:
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = a["x"] + a["width"], a["y"] + a["height"]

    bx1, by1 = b["x"], b["y"]
    bx2, by2 = b["x"] + b["width"], b["y"] + b["height"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(1, a["width"] * a["height"])
    area_b = max(1, b["width"] * b["height"])

    return inter_area / float(area_a + area_b - inter_area)


def x_overlap_ratio(a: dict, b: dict) -> float:
    ax1, ax2 = a["x"], a["x"] + a["width"]
    bx1, bx2 = b["x"], b["x"] + b["width"]

    inter = max(0, min(ax2, bx2) - max(ax1, bx1))
    denom = max(1, min(a["width"], b["width"]))

    return inter / denom


def normalize_line_box(x: int, y_center: int, width: int, img_w: int, img_h: int, source: str):
    guide_h = max(18, int(img_h * 0.03))
    x = max(0, int(x))
    y = max(0, int(y_center - guide_h // 2))
    width = min(int(width), img_w - x)
    height = min(guide_h, img_h - y)

    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "source_methods": [source],
    }


def is_valid_line_box(box: dict, img_w: int, img_h: int) -> bool:
    w = box["width"]
    h = box["height"]
    x = box["x"]
    y = box["y"]

    if w < img_w * 0.07:
        return False

    if w > img_w * 0.65:
        return False

    if h > img_h * 0.12:
        return False

    if y < img_h * 0.04:
        return False

    if y > img_h * 0.96:
        return False

    if x < 0 or x >= img_w:
        return False

    return True


def merge_boxes(boxes: list, img_w: int, img_h: int) -> list:
    """
    NMS/유사 좌표 기반 병합.
    같은 입력선을 여러 알고리즘이 중복 탐지한 경우 하나로 합침.
    """
    if not boxes:
        return []

    boxes = [b for b in boxes if is_valid_line_box(b, img_w, img_h)]
    boxes = sorted(boxes, key=lambda b: (b["y"] + b["height"] // 2, b["x"]))

    merged = []

    y_tol = max(10, int(img_h * 0.018))

    for box in boxes:
        bx_center_y = box["y"] + box["height"] // 2
        merged_into_existing = False

        for m in merged:
            my_center_y = m["y"] + m["height"] // 2
            same_y = abs(bx_center_y - my_center_y) <= y_tol
            iou = box_iou(box, m)
            overlap = x_overlap_ratio(box, m)

            if iou > 0.12 or (same_y and overlap > 0.35):
                x1 = min(m["x"], box["x"])
                y1 = min(m["y"], box["y"])
                x2 = max(m["x"] + m["width"], box["x"] + box["width"])
                y2 = max(m["y"] + m["height"], box["y"] + box["height"])

                m["x"] = x1
                m["y"] = y1
                m["width"] = x2 - x1
                m["height"] = y2 - y1
                m["source_methods"] = sorted(
                    list(set(m.get("source_methods", []) + box.get("source_methods", [])))
                )
                merged_into_existing = True
                break

        if not merged_into_existing:
            merged.append(box)

    merged = [b for b in merged if is_valid_line_box(b, img_w, img_h)]
    merged = sorted(merged, key=lambda b: (b["y"] + b["height"] // 2, b["x"]))

    return merged


# =========================================================
# OpenCV 후보 탐지 앙상블
# =========================================================
def detect_hough_lines(gray: np.ndarray, img_w: int, img_h: int) -> list:
    boxes = []

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 120)

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(35, img_w // 35), 1)
    )
    horizontal = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)

    lines = cv2.HoughLinesP(
        horizontal,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=max(60, int(img_w * 0.07)),
        maxLineGap=max(10, int(img_w * 0.025)),
    )

    if lines is None:
        return boxes

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if abs(y1 - y2) > 6:
            continue

        x_left = min(x1, x2)
        x_right = max(x1, x2)
        y_center = int((y1 + y2) / 2)
        width = x_right - x_left

        box = normalize_line_box(x_left, y_center, width, img_w, img_h, "hough")
        boxes.append(box)

    return boxes


def detect_adaptive_morphology(gray: np.ndarray, img_w: int, img_h: int) -> list:
    boxes = []

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(45, img_w // 28), 1)
    )

    horizontal = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, horizontal_kernel)
    horizontal = cv2.dilate(horizontal, np.ones((1, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(
        horizontal,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        y_center = y + h // 2

        box = normalize_line_box(x, y_center, w, img_w, img_h, "adaptive_morph")
        boxes.append(box)

    return boxes


def detect_contour_blackhat(gray: np.ndarray, img_w: int, img_h: int) -> list:
    boxes = []

    blackhat_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(35, img_w // 32), 3)
    )
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, blackhat_kernel)

    _, thresh = cv2.threshold(blackhat, 8, 255, cv2.THRESH_BINARY)

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(40, img_w // 30), 1)
    )
    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel)
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, horizontal_kernel)

    contours, _ = cv2.findContours(
        horizontal,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        y_center = y + h // 2

        box = normalize_line_box(x, y_center, w, img_w, img_h, "contour_blackhat")
        boxes.append(box)

    return boxes


def detect_input_line_candidates(image: Image.Image) -> list:
    """
    3가지 방식 앙상블:
    A. HoughLinesP 기반 선 탐지
    B. Adaptive Threshold + Horizontal Morphology
    C. Contour + Blackhat 기반 탐지

    이후 유사 좌표 병합으로 최종 후보 생성.
    """
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    img = np.array(rgb)
    img_h, img_w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    boxes = []
    boxes.extend(detect_hough_lines(gray, img_w, img_h))
    boxes.extend(detect_adaptive_morphology(gray, img_w, img_h))
    boxes.extend(detect_contour_blackhat(gray, img_w, img_h))

    merged = merge_boxes(boxes, img_w, img_h)

    candidates = []

    for idx, box in enumerate(merged, start=1):
        x = box["x"]
        y = box["y"]
        w = box["width"]
        h = box["height"]

        cx = int(x + w / 2)
        cy = int(y + h / 2)

        candidates.append(
            {
                "candidate_id": idx,
                "pixel_center": {
                    "x": cx,
                    "y": cy,
                },
                "pixel_bbox": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                },
                "center": {
                    "x_percent": round(cx / img_w * 100, 2),
                    "y_percent": round(cy / img_h * 100, 2),
                },
                "bounding_box": {
                    "x_percent": round(x / img_w * 100, 2),
                    "y_percent": round(y / img_h * 100, 2),
                    "width_percent": round(w / img_w * 100, 2),
                    "height_percent": round(h / img_h * 100, 2),
                },
                "source_methods": box.get("source_methods", []),
            }
        )

    return candidates[:40]


# =========================================================
# 시각화
# =========================================================
def get_font(size: int = 18):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_candidate_overlay(image: Image.Image, candidates: list) -> Image.Image:
    annotated = ImageOps.exif_transpose(image).convert("RGB")
    draw = ImageDraw.Draw(annotated)
    font = get_font(18)

    for c in candidates:
        cid = c["candidate_id"]
        box = c["pixel_bbox"]

        left = box["x"]
        top = box["y"]
        right = box["x"] + box["width"]
        bottom = box["y"] + box["height"]

        for thickness in range(3):
            draw.rectangle(
                [left - thickness, top - thickness, right + thickness, bottom + thickness],
                outline="blue",
            )

        label = f"C{cid}"
        label_y = max(0, top - 22)
        text_bbox = draw.textbbox((left, label_y), label, font=font)

        pad = 4
        bg = [
            text_bbox[0] - pad,
            text_bbox[1] - pad,
            text_bbox[2] + pad,
            text_bbox[3] + pad,
        ]

        draw.rectangle(bg, fill="blue")
        draw.text((left, label_y), label, fill="white", font=font)

    return annotated


def draw_final_overlay(image: Image.Image, fields: list, system_valid: bool) -> Image.Image:
    annotated = ImageOps.exif_transpose(image).convert("RGB")
    draw = ImageDraw.Draw(annotated)
    img_w, img_h = annotated.size
    font = get_font(18)

    color = "red" if system_valid else "orange"

    for item in fields:
        if not item.get("found", False):
            continue

        field_name = item.get("field_name", "Unknown")
        center = item.get("center", {})

        cx_percent = clamp_percent(center.get("x_percent", 0))
        cy_percent = clamp_percent(center.get("y_percent", 0))

        cx = int(img_w * cx_percent / 100)
        cy = int(img_h * cy_percent / 100)

        box_w = int(img_w * 0.22)
        box_h = int(img_h * 0.045)

        left = max(0, cx - box_w // 2)
        top = max(0, cy - box_h // 2)
        right = min(img_w - 1, cx + box_w // 2)
        bottom = min(img_h - 1, cy + box_h // 2)

        for thickness in range(4):
            draw.rectangle(
                [left - thickness, top - thickness, right + thickness, bottom + thickness],
                outline=color,
            )

        radius = 6
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=color,
            outline="white",
            width=2,
        )

        label = field_name
        label_x = left
        label_y = max(0, top - 24)

        text_bbox = draw.textbbox((label_x, label_y), label, font=font)
        pad = 4

        bg_box = [
            text_bbox[0] - pad,
            text_bbox[1] - pad,
            text_bbox[2] + pad,
            text_bbox[3] + pad,
        ]

        draw.rectangle(bg_box, fill=color)
        draw.text((label_x, label_y), label, fill="white", font=font)

    return annotated


# =========================================================
# GPT 응답 정규화
# =========================================================
def normalize_fields(fields: list) -> list:
    normalized = []

    by_name = {}
    for item in fields:
        name = str(item.get("field_name", "")).strip()
        by_name[name.lower()] = item

    for target in TARGET_FIELDS:
        item = by_name.get(target.lower())

        if item is None:
            item = {
                "field_name": target,
                "found": False,
                "candidate_id": None,
                "confidence": 0,
                "visual_evidence": "Not found in model response.",
                "mouse_action_plan": [
                    {"step": 1, "action": "Field was not found. Skip automation for this field."}
                ],
            }

        item["field_name"] = target
        item.setdefault("found", False)
        item.setdefault("candidate_id", None)
        item.setdefault("confidence", 0)
        item.setdefault("visual_evidence", "")
        item.setdefault("mouse_action_plan", [])

        item["candidate_id"] = parse_candidate_id(item.get("candidate_id"))
        item["model_confidence"] = clamp_confidence(item.get("confidence", 0))
        item["confidence"] = item["model_confidence"]

        normalized.append(item)

    return normalized


def validate_matching(fields: list, candidates: list) -> dict:
    """
    시스템 자체 검증 레이어.
    GPT confidence와 별개로 자동화 가능 여부를 판단한다.
    """
    errors = []
    valid_candidate_ids = {c["candidate_id"] for c in candidates}

    used = {}

    for item in fields:
        field_name = item.get("field_name")
        found = item.get("found", False)
        cid = item.get("candidate_id")

        if not found:
            errors.append(f"{field_name}: found=false 상태입니다.")
            continue

        if cid is None:
            errors.append(f"{field_name}: candidate_id가 없습니다.")
            continue

        if cid not in valid_candidate_ids:
            errors.append(
                f"{field_name}: candidate_id={cid}는 OpenCV 후보 목록에 존재하지 않습니다."
            )
            continue

        used.setdefault(cid, []).append(field_name)

    for cid, field_names in used.items():
        if len(field_names) > 1:
            errors.append(
                f"candidate_id={cid}가 여러 필드에 중복 할당되었습니다: {', '.join(field_names)}"
            )

    unique_assigned = set(used.keys())

    if len(unique_assigned) != len(TARGET_FIELDS):
        errors.append(
            f"7개 필드가 7개의 고유 후보에 매칭되지 않았습니다. "
            f"현재 고유 후보 수: {len(unique_assigned)}"
        )

    return {
        "system_valid": len(errors) == 0,
        "validation_errors": errors,
    }


def enrich_fields_with_candidates(fields: list, candidates: list, validation: dict) -> list:
    candidate_map = {c["candidate_id"]: c for c in candidates}
    system_valid = validation["system_valid"]

    for item in fields:
        cid = item.get("candidate_id")
        c = candidate_map.get(cid)

        item["system_valid"] = system_valid
        item.setdefault("validation_error", "")

        if item.get("found", False) and c is not None:
            item["element_type"] = "text_input"
            item["center"] = c["center"]
            item["bounding_box"] = c["bounding_box"]
            item["pixel_center"] = c["pixel_center"]
            item["pixel_bbox"] = c["pixel_bbox"]

            item["mouse_action_plan"] = [
                {
                    "step": 1,
                    "action": f"Move mouse to candidate C{cid} center at pixel ({c['pixel_center']['x']}, {c['pixel_center']['y']})."
                },
                {
                    "step": 2,
                    "action": "Click the input line."
                },
                {
                    "step": 3,
                    "action": f"Type the corresponding value for {item['field_name']}."
                },
            ]
        else:
            item["found"] = False
            item["element_type"] = "text_input"
            item["center"] = {"x_percent": 0, "y_percent": 0}
            item["bounding_box"] = {
                "x_percent": 0,
                "y_percent": 0,
                "width_percent": 0,
                "height_percent": 0,
            }
            item["pixel_center"] = {"x": 0, "y": 0}
            item["pixel_bbox"] = {"x": 0, "y": 0, "width": 0, "height": 0}
            item["mouse_action_plan"] = [
                {
                    "step": 1,
                    "action": "Field is not safe for automation because it failed validation."
                }
            ]

    return fields


# =========================================================
# OpenAI API 호출
# =========================================================
def analyze_rpa_challenge(api_key: str, image: Image.Image, model_name: str):
    """
    전체 파이프라인:
    1. OpenCV 3방식 앙상블로 후보 탐지
    2. 후보 수 7개 미만이면 GPT 호출 중단
    3. 후보 이미지를 GPT에게 전달
    4. GPT가 field_name -> candidate_id 매칭
    5. 시스템 자체 검증
    6. 검증 통과 시 최종 좌표 사용
    """
    candidates = detect_input_line_candidates(image)

    if len(candidates) < len(TARGET_FIELDS):
        raise ValueError(
            f"OpenCV 입력선 후보가 부족합니다. "
            f"필요 후보 수: {len(TARGET_FIELDS)}개, 탐지 후보 수: {len(candidates)}개. "
            f"이미지를 더 선명하게 캡처하거나, RPA Challenge 입력 폼 영역이 충분히 보이도록 업로드해 주세요."
        )

    candidate_overlay = draw_candidate_overlay(image, candidates)
    candidate_overlay_b64 = image_to_base64_png(candidate_overlay)
    user_prompt = build_user_prompt(candidates)

    client = OpenAI(api_key=api_key)

    completion = client.chat.completions.create(
        model=model_name,
        temperature=0,
        max_tokens=1800,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{candidate_overlay_b64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
    )

    raw_text = completion.choices[0].message.content

    gpt_fields = extract_json_array(raw_text)
    normalized_fields = normalize_fields(gpt_fields)
    validation = validate_matching(normalized_fields, candidates)
    final_fields = enrich_fields_with_candidates(normalized_fields, candidates, validation)

    analysis_result = {
        "system_valid": validation["system_valid"],
        "validation_errors": validation["validation_errors"],
        "candidate_count": len(candidates),
        "valid_candidate_ids": [c["candidate_id"] for c in candidates],
        "fields": final_fields,
    }

    return analysis_result, raw_text, candidates, candidate_overlay


# =========================================================
# UI
# =========================================================
st.markdown(
    "<h1>🤖 RPA Challenge VLM Agent</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    RPA Challenge 화면 이미지를 업로드하면  
    **OpenCV가 입력선 후보를 먼저 찾고**, GPT-4o Vision이  
    7개 필드를 후보 번호와 매칭합니다.  
    이후 프로그램이 GPT 결과를 검증하여 자동화 가능 여부를 판단합니다.
    """
)

st.info(
    "이번 버전은 'AI가 맞다고 하면 믿는 구조'가 아니라, "
    "OpenCV 후보 탐지 + GPT 후보 매칭 + 시스템 자체 검증 레이어를 가진 구조입니다."
)


# =========================================================
# 사이드바
# =========================================================
with st.sidebar:
    st.header("🔑 OpenAI 설정")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-..."
    )

    model_name = st.selectbox(
        "모델 선택",
        options=["gpt-4o", "gpt-4o-mini"],
        index=0,
    )

    st.divider()

    st.header("📌 분석 대상 필드")
    for field in TARGET_FIELDS:
        st.write(f"- {field}")

    st.divider()

    st.caption(
        "API Key는 이 앱 실행 중 메모리에서만 사용됩니다. "
        "GitHub에 키를 올리지 않도록 주의하세요."
    )


# =========================================================
# 이미지 업로드
# =========================================================
st.subheader("1. RPA Challenge 화면 이미지 업로드")

uploaded_file = st.file_uploader(
    "RPA Challenge 화면 캡처 이미지를 업로드하세요.",
    type=["png", "jpg", "jpeg", "webp"],
)

image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")

    st.image(
        image,
        caption=f"업로드된 이미지: {uploaded_file.name}",
        use_container_width=True,
    )
else:
    st.warning("먼저 RPA Challenge 화면 캡처 이미지를 업로드해줘.")


# =========================================================
# 분석 버튼
# =========================================================
st.subheader("2. 자동 분석 실행")

analyze_button = st.button(
    "🚀 RPA Challenge 화면 분석하기",
    use_container_width=True,
)

if analyze_button:
    if not api_key:
        st.error("사이드바에 OpenAI API Key를 입력해줘.")
    elif image is None:
        st.error("분석할 RPA Challenge 화면 이미지를 먼저 업로드해줘.")
    else:
        with st.spinner("OpenCV 후보 탐지 + GPT-4o 필드 매칭 + 시스템 검증을 실행하는 중입니다..."):
            try:
                analysis_result, raw_text, candidates, candidate_overlay = analyze_rpa_challenge(
                    api_key=api_key,
                    image=image,
                    model_name=model_name,
                )

                st.session_state["analysis_result"] = analysis_result
                st.session_state["raw_text"] = raw_text
                st.session_state["candidates"] = candidates
                st.session_state["candidate_overlay"] = candidate_overlay
                st.session_state["analyzed_at"] = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                if analysis_result["system_valid"]:
                    st.success("분석 완료! 시스템 검증을 통과했습니다.")
                else:
                    st.error("분석은 완료되었지만 시스템 검증에 실패했습니다. 자동화는 중단해야 합니다.")

            except Exception as e:
                st.error("분석 중 오류가 발생했습니다.")
                st.exception(e)


# =========================================================
# 결과 표시
# =========================================================
if "analysis_result" in st.session_state and image is not None:
    analysis_result = st.session_state["analysis_result"]
    fields = analysis_result["fields"]
    system_valid = analysis_result["system_valid"]
    validation_errors = analysis_result["validation_errors"]

    candidates = st.session_state.get("candidates", [])
    candidate_overlay = st.session_state.get("candidate_overlay", None)

    st.divider()

    st.subheader("3. 시스템 검증 결과")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("System Valid", "TRUE" if system_valid else "FALSE")

    with col2:
        st.metric("Candidate Count", analysis_result.get("candidate_count", 0))

    with col3:
        st.metric("Required Fields", len(TARGET_FIELDS))

    if system_valid:
        st.success("7개 필드가 모두 고유하고 유효한 후보에 매칭되었습니다.")
    else:
        st.error("검증 실패: 자동화 실행 전 반드시 수정이 필요합니다.")
        for err in validation_errors:
            st.warning(err)

    st.subheader("4. OpenCV 입력선 후보 탐지 결과")

    if candidate_overlay is not None:
        st.image(
            candidate_overlay,
            caption="OpenCV 3방식 앙상블으로 탐지한 입력선 후보",
            use_container_width=True,
        )

    with st.expander("OpenCV 후보 좌표 보기"):
        st.json(candidates)

    st.subheader("5. 최종 좌표 시각화 결과")

    if system_valid:
        final_overlay = draw_final_overlay(image, fields, system_valid=True)
        st.image(
            final_overlay,
            caption="검증 통과: 최종 자동화 좌표",
            use_container_width=True,
        )
    else:
        st.warning("시스템 검증에 실패했기 때문에 최종 자동화 좌표 시각화는 참고용으로만 표시합니다.")
        final_overlay = draw_final_overlay(image, fields, system_valid=False)
        st.image(
            final_overlay,
            caption="검증 실패: 참고용 좌표",
            use_container_width=True,
        )

    st.subheader("6. 필드별 탐지 결과")

    for item in fields:
        found = item.get("found", False)
        field_name = item.get("field_name", "-")
        confidence = item.get("confidence", 0)
        candidate_id = item.get("candidate_id", None)
        center = item.get("center", {})
        pixel_center = item.get("pixel_center", {})
        box = item.get("bounding_box", {})

        with st.expander(
            f"{'✅' if found else '❌'} {field_name} | candidate: {candidate_id} | model_confidence: {confidence}"
        ):
            col_a, col_b = st.columns(2)

            with col_a:
                st.write("**탐지 여부:**", found)
                st.write("**후보 ID:**", candidate_id)
                st.write("**시스템 검증:**", item.get("system_valid", False))
                st.write("**요소 타입:**", item.get("element_type", "-"))
                st.write("**시각적 근거:**", item.get("visual_evidence", "-"))

            with col_b:
                st.write("**Center 좌표(%)**")
                st.code(
                    json.dumps(center, ensure_ascii=False, indent=2),
                    language="json",
                )

                st.write("**Pixel Center**")
                st.code(
                    json.dumps(pixel_center, ensure_ascii=False, indent=2),
                    language="json",
                )

                st.write("**Bounding Box(%)**")
                st.code(
                    json.dumps(box, ensure_ascii=False, indent=2),
                    language="json",
                )

            st.write("**마우스 클릭 액션 플랜**")
            for step in item.get("mouse_action_plan", []):
                st.write(f"- Step {step.get('step')}: {step.get('action')}")

    st.subheader("7. 최종 JSON 결과")

    st.json(analysis_result)

    json_data = json.dumps(analysis_result, ensure_ascii=False, indent=2)

    st.download_button(
        label="📥 JSON 결과 다운로드",
        data=json_data,
        file_name="rpa_challenge_validated_result.json",
        mime="application/json",
        use_container_width=True,
    )

    with st.expander("GPT 모델 원본 응답 보기"):
        st.code(st.session_state.get("raw_text", ""), language="json")


# =========================================================
# 하단 안내
# =========================================================
st.divider()

st.markdown(
    """
    ### 다음 단계

    이번 버전은 **OpenCV 후보 탐지 + GPT 후보 매칭 + 시스템 자체 검증** 구조입니다.

    다음에는 아래 기능으로 확장할 수 있습니다.

    - CSV/엑셀 데이터 읽기
    - 검증 통과 시에만 PyAutoGUI 클릭 실행
    - 필드별 값 자동 입력
    - Submit 버튼 자동 클릭
    - 화면 재배치 후 재캡처 및 재분석
    - 후보 탐지 실패 시 자동 재시도 또는 이미지 crop 전처리
    """
)