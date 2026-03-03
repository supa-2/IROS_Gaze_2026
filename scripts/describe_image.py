#!/usr/bin/env python3
import base64
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Load image
with open('data/G.jpg', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

client = OpenAI(
    api_key=os.getenv('QWEN_API_KEY'),
    base_url=os.getenv('QWEN_BASE_URL')
)

# Ask VLM to describe what it sees
prompt = """Describe this image in detail:
1. What objects are visible?
2. Where are they located? (approximate positions)
3. What are the main elements at the TOP of the image?
4. What are the main elements at the BOTTOM of the image?
5. Are there paintings? If yes, where are they located?

Use coordinate system: (0,0) is top-left, y increases downward."""

response = client.chat.completions.create(
    model='qwen-vl-max-latest',
    messages=[
        {'role': 'user', 'content': [
            {'type': 'text', 'text': prompt},
            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_base64}'}}
        ]}
    ],
    temperature=0.1,
    max_tokens=500
)

print(response.choices[0].message.content)
