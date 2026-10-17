import os
import json
import re
import subprocess
import streamlit as st

@st.cache_resource
def ensure_packages():
    try:
        subprocess.run(["pip", "install", "yt-dlp", "opencv-python-headless"], check=True)
    except Exception as e:
        st.warning(f"Package setup note: {e}")

ensure_packages()

# REMOVED top-level imports of cv2 and yt_dlp to prevent startup crashes

from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

st.set_page_config(page_title="AI Automotive Benchmarking", layout="wide")

st.title("🚗 AI Automotive Channel-Wide Benchmarking")
st.markdown("Query the channel-wide pre-indexed RAG vector store (**13,850+ chunks across 97 videos**) with direct stream frame extraction.")

WORKFLOW_FILE = "youtube_workflow.json"

tab_run, tab_studio = st.tabs(["🚀 Query Vector DB & Benchmark", "🛠️ LfD Workflow Studio (Schema Inspector)"])

def extract_exact_frame(video_id, start_sec, output_path):
    """
    Bypasses YouTube bot walls entirely by pulling the direct stream URL via yt-dlp
    and extracting the exact frame at start_sec using OpenCV.
    """
    try:
        # Lazy import inside the function so packages are guaranteed to be installed first
        import cv2
        import yt_dlp

        ydl_opts = {'format': 'best[height<=720]', 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            stream_url = info.get('url')
            if not stream_url:
                for f in info.get('formats', []):
                    if f.get('url') and f.get('vcodec') != 'none' and f.get('acodec') != 'none':
                        stream_url = f['url']
                        break
        
        if not stream_url:
            return False

        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            return False

        # Seek to exact millisecond timestamp
        cap.set(cv2.CAP_PROP_POS_MSEC, float(start_sec) * 1000.0)
        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            cv2.imwrite(output_path, frame)
            return True
    except Exception as e:
        print(f"Frame extraction error: {e}")
    return False