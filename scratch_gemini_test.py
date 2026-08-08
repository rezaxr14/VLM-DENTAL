import os
from PIL import Image
from google import genai
from google.genai import types
from dental_agent.config import load_env

load_env()

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    keys_raw = os.environ.get("GEMINI_API_KEYS", "")
    api_key = [k.strip().strip("'\"") for k in keys_raw.split(",") if k.strip().strip("'\"")][0]

client = genai.Client(api_key=api_key)

img = Image.new("RGB", (100, 100), color="red")

try:
    content = types.Content(role="user", parts=[
        types.Part.from_text("Analyze this:"), 
        img
    ])
    print("Content creation succeeded with Image directly in parts.")
except Exception as e:
    print(f"Content creation failed: {e}")

try:
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = buf.getvalue()
    
    content = types.Content(role="user", parts=[
        types.Part.from_text("Analyze this:"), 
        types.Part.from_bytes(data=b64, mime_type="image/jpeg")
    ])
    print("Content creation succeeded with from_bytes.")
except Exception as e:
    print(f"from_bytes creation failed: {e}")
