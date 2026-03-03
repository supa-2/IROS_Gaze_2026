#!/usr/bin/env python3
import base64
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import re

load_dotenv()

# Load image
with open('data/G.jpg', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

# Get current bboxes
current_bboxes = [
    {'id': 'E1', 'bbox': [79, 473, 179, 581]},
    {'id': 'E2', 'bbox': [208, 447, 339, 567]},
    {'id': 'E3', 'bbox': [397, 457, 507, 602]},
    {'id': 'E4', 'bbox': [613, 446, 792, 571]},
    {'id': 'E5', 'bbox': [859, 461, 968, 602]}
]

# Format current bboxes for the prompt
bbox_str = '\n'.join([f"{ex['id']}: {ex['bbox']}" for ex in current_bboxes])

client = OpenAI(
    api_key=os.getenv('QWEN_API_KEY'),
    base_url=os.getenv('QWEN_BASE_URL')
)

prompt = f"""I need you to verify and correct bounding box coordinates for paintings in this image.

Image size: 1100 pixels wide, 618 pixels high
Coordinate system: (0,0) is TOP-LEFT corner, x goes right, y goes down

Current bounding boxes (may be incorrect):
{bbox_str}

Please:
1. Look at the image and identify all 5 paintings
2. Check if each current bbox correctly covers its painting
3. If a bbox is wrong, provide the CORRECT coordinates

Return ONLY valid JSON format:
[
  {{"id": "E1", "correct": true, "bbox": [x1, y1, x2, y2], "note": "correct"}},
  {{"id": "E2", "correct": false, "bbox": [NEW_x1, NEW_y1, NEW_x2, NEW_y2], "note": "was too far left"}},
  ...
]

Where:
- id: the exhibit ID
- correct: true if bbox is accurate, false if needs correction
- bbox: [x1, y1, x2, y2] - top-left and bottom-right coordinates
- note: brief explanation

IMPORTANT: Make sure each bbox TIGHTLY encloses the painting content including its frame!"""

print('[*] Asking VLM to verify and correct bboxes...')

response = client.chat.completions.create(
    model='qwen-vl-max-latest',
    messages=[
        {'role': 'user', 'content': [
            {'type': 'text', 'text': prompt},
            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_base64}'}}
        ]}
    ],
    temperature=0.1,
    max_tokens=1500
)

result = response.choices[0].message.content
print('[+] VLM response:')
print(result)

# Parse JSON
json_match = re.search(r'\[.*\]', result, re.DOTALL)
if json_match:
    try:
        corrected = json.loads(json_match.group())
        print('\n[*] Corrected bboxes:')
        for item in corrected:
            print(f"  {item['id']}: correct={item.get('correct')}, bbox={item['bbox']}, note={item.get('note', '')}")

        # Save corrected bboxes
        os.makedirs('data/outputs/G', exist_ok=True)
        with open('data/outputs/G/corrected_bboxes.json', 'w') as f:
            json.dump(corrected, f, indent=2)
        print('\n[+] Saved: data/outputs/G/corrected_bboxes.json')
    except json.JSONDecodeError as e:
        print(f'[!] JSON parse error: {e}')
else:
    print('[!] No JSON found in response')
