import os
from dotenv import load_dotenv
import asyncio

def load_api_key():
    """Load Google Gemini API key from .env file or environment variable."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment.")
    return api_key

def save_uploaded_file(uploaded_file, save_dir="temp_files"):
    """Save uploaded file to a temporary directory and return the path."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop()) 