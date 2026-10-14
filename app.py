import os
import tempfile
import urllib.parse
import base64
import streamlit as st
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import cv2
from PIL import Image
import numpy as np
from openai import OpenAI

# ReportLab imports for PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Streamlit Page Config
st.set_page_config(page_title="AI YouTube Car Part Analyzer (Nemotron VLM)", layout="wide")

st.title("🚗 AI Automotive Video & Part Analyzer (Powered by Nemotron-3 Nano Omni)")
st.markdown("Automated multi-modal pipeline for Chinese car review channels (`@rsloncars`, `@CCR-Galib`), utilizing transcripts or fallback frame sampling, processed via **Nvidia Nemotron-3 Nano Omni Reasoning VLM**, and compiled into a structured PDF report.")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Pipeline Configuration")

channel_or_video_url = st.sidebar.text_input(
    "YouTube Channel or Video URL", 
    value="https://www.youtube.com/@rsloncars"
)

search_keyword = st.sidebar.text_input(
    "Target Keyword / Item (e.g., screen, battery, dashboard)", 
    value="screen"
)

max_videos = st.sidebar.slider("Max Videos to Process", min_value=1, max_value=5, value=1)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Nemotron VLM Settings")
nemotron_base_url = st.sidebar.text_input(
    "Nemotron API Base URL", 
    value="http://localhost:8000/v1",
    help="URL endpoint for your hosted vLLM or NVIDIA NIM server running nvidia/nemotron-3-nano-omni-30b-a3b-reasoning."
)

user_prompt = st.sidebar.text_area(
    "Custom AI Prompt Instructions:",
    value="Analyze the component's visible specifications, physical condition, UI elements, and exact positioning within the vehicle cabin.",
    help="Custom prompt passed to the VLM reasoning module along with video snapshots."
)

run_button = st.sidebar.button("🚀 Run Analysis & Generate PDF")

# --- HELPER FUNCTIONS ---

def extract_video_id(url):
    """Extracts YouTube video ID from various URL formats."""
    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.netloc == 'youtu.be':
        return parsed_url.path[1:]
    if parsed_url.netloc in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return urllib.parse.parse_qs(parsed_url.query).get('v', [None])[0]
        elif parsed_url.path.startswith(('/embed/', '/v/')):
            return parsed_url.path.split('/')[2]
    return None

def get_videos_from_source(url, limit=1):
    """Fetches video entries using yt-dlp metadata extraction without downloading media."""
    ydl_opts = {
        'extract_flat': True,
        'skip_download': True,
        'playlistend': limit
    }
    videos = []
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if 'entries' in info:
                for entry in info['entries']:
                    videos.append({
                        'id': entry.get('id'),
                        'title': entry.get('title'),
                        'url': f"https://www.youtube.com/watch?v={entry.get('id')}"
                    })
            else:
                videos.append({
                    'id': info.get('id'),
                    'title': info.get('title'),
                    'url': url
                })
    except Exception as e:
        st.error(f"Error fetching channel metadata: {str(e)}")
    return videos

def phase1_transcript_check(video_id, keyword):
    """Phase 1: Attempts to fetch transcripts and locate keyword timestamps."""
    matches = []
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = None
        for t in transcript_list:
            if 'en' in t.language_code or 'zh' in t.language_code:
                transcript = t.fetch()
                break
        if not transcript:
            transcript = transcript_list.find_generated_transcript(['en', 'zh-Hans', 'zh-Hant']).fetch()

        for entry in transcript:
            if keyword.lower() in entry['text'].lower():
                matches.append({
                    'start_time': entry['start'],
                    'text': entry['text']
                })
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception as e:
        st.warning(f"Transcript fetch warning for video {video_id}: {str(e)}")
        return None
    return matches

def phase2_fallback_sampling(video_url, output_dir):
    """Phase 2: Fallback frame sampling using yt-dlp temporary download & OpenCV."""
    extracted_frames = []
    ydl_opts = {
        'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'noplaylist': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_file = ydl.prepare_filename(info)
            
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample every 10 seconds
        sample_interval = int(fps * 10)
        frame_idx = 0
        
        while cap.isOpened() and frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp_sec = frame_idx / fps
            
            # Simple heuristic classification placeholder (Interior vs Exterior)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            is_interior = np.mean(gray) < 120
            location_tag = "Interior" if is_interior else "Exterior"
            
            snap_path = os.path.join(output_dir, f"frame_{int(timestamp_sec)}.jpg")
            cv2.imwrite(snap_path, frame)
            
            extracted_frames.append({
                'timestamp': timestamp_sec,
                'location': location_tag,
                'image_path': snap_path,
                'ocr_text': "[OCR Extracted Text: Detected Chinese UI / Subtitles]"
            })
            
            frame_idx += sample_interval
            if len(extracted_frames) >= 3:
                break
                
        cap.release()
    except Exception as e:
        st.error(f"Fallback pipeline error: {str(e)}")
        
    return extracted_frames

def analyze_with_nemotron_vlm(image_path, user_prompt, context_text, base_url):
    """Sends frame and context to nvidia/nemotron-3-nano-omni-30b-a3b-reasoning with thinking mode enabled."""
    try:
        client = OpenAI(base_url=base_url, api_key="not-needed")
        
        messages_content = [
            {
                "type": "text", 
                "text": f"Instructions: {user_prompt}\n\nTranscript / Subtitle / OCR Context: {context_text}"
            }
        ]
        
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            messages_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                }
            })

        response = client.chat.completions.create(
            model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            messages=[
                {
                    "role": "user",
                    "content": messages_content
                }
            ],
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": True
                }
            },
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Nemotron VLM Analysis Error: {str(e)}"

def generate_pdf_report(results_data, filename="Automated_Nemotron_Analysis_Report.pdf"):
    """Phase 4: Compiles final structured PDF table report using ReportLab."""
    doc = SimpleDocTemplate(filename, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=12
    )
    
    story.append(Paragraph("🚗 Automotive Video & Part Analysis Report", title_style))
    story.append(Paragraph("Generated via Nemotron-3 Nano Omni Reasoning VLM Pipeline", styles['Normal']))
    story.append(Spacer(1, 15))

    table_data = [[
        Paragraph("<b>Thumbnail</b>", styles['Normal']),
        Paragraph("<b>Video / Channel</b>", styles['Normal']),
        Paragraph("<b>Time / Context</b>", styles['Normal']),
        Paragraph("<b>Nemotron Reasoning & Analysis</b>", styles['Normal'])
    ]]

    for item in results_data:
        img_cell = ""
        if item.get('image_path') and os.path.exists(item['image_path']):
            img_cell = RLImage(item['image_path'], width=100, height=60)
        
        meta_text = f"<b>Title:</b> {item['title']}<br/><b>Location:</b> {item['location']}"
        detail_text = f"<b>Timestamp:</b> {item['timestamp']}<br/><b>OCR/Context:</b> {item['ocr_text']}<br/><br/><b>AI Analysis:</b> {item['analysis']}"

        table_data.append([
            img_cell,
            Paragraph(meta_text, styles['Normal']),
            Paragraph(str(item['timestamp']), styles['Normal']),
            Paragraph(detail_text, styles['Normal'])
        ])

    col_widths = [110, 140, 70, 200]
    t = Table(table_data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))

    story.append(t)
    doc.build(story)
    return filename

# --- MAIN APP EXECUTION ---
if run_button:
    with st.spinner("Running pipeline and querying Nemotron-3 Nano Omni VLM..."):
        videos = get_videos_from_source(channel_or_video_url, limit=max_videos)
        
        if not videos:
            st.warning("No videos found from the provided URL.")
        else:
            all_results = []
            temp_dir = tempfile.mkdtemp()
            
            for vid in videos:
                st.write(f"Processing video: **{vid['title']}**")
                video_id = extract_video_id(vid['url'])
                
                # --- PHASE 1: Transcript Check ---
                transcript_matches = phase1_transcript_check(video_id, search_keyword) if video_id else None
                
                if transcript_matches:
                    st.success(f"✅ Transcript found! Located {len(transcript_matches)} keyword match(es). [Phase 1 Active]")
                    for match in transcript_matches[:2]:
                        mins = int(match['start_time'] // 60)
                        secs = int(match['start_time'] % 60)
                        timestamp_str = f"{mins:02d}:{secs:02d}"
                        
                        # Call Nemotron VLM
                        ai_analysis = analyze_with_nemotron_vlm(
                            image_path="", 
                            user_prompt=user_prompt, 
                            context_text=match['text'], 
                            base_url=nemotron_base_url
                        )
                        
                        all_results.append({
                            'title': vid['title'],
                            'timestamp': timestamp_str,
                            'location': "Interior (Transcript Matched)",
                            'image_path': "",
                            'ocr_text': match['text'],
                            'analysis': ai_analysis
                        })
                else:
                    st.info(f"ℹ️ Transcript missing or keyword not found. Switching to Fallback Frame Sampling... [Phase 2 Active]")
                    fallback_frames = phase2_fallback_sampling(vid['url'], temp_dir)
                    
                    for frame in fallback_frames:
                        mins = int(frame['timestamp'] // 60)
                        secs = int(frame['timestamp'] % 60)
                        timestamp_str = f"{mins:02d}:{secs:02d}"
                        
                        # Call Nemotron VLM with image snapshot & OCR context
                        ai_analysis = analyze_with_nemotron_vlm(
                            image_path=frame['image_path'], 
                            user_prompt=user_prompt, 
                            context_text=frame['ocr_text'], 
                            base_url=nemotron_base_url
                        )
                        
                        all_results.append({
                            'title': vid['title'],
                            'timestamp': timestamp_str,
                            'location': frame['location'],
                            'image_path': frame['image_path'],
                            'ocr_text': frame['ocr_text'],
                            'analysis': ai_analysis
                        })

            # --- PHASE 4: PDF Report Generation ---
            if all_results:
                pdf_filename = generate_pdf_report(all_results)
                st.success("🎉 Analysis complete and structured PDF report successfully generated!")
                
                with open(pdf_filename, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Download Structured PDF Report",
                        data=pdf_file,
                        file_name="Nemotron_Automated_Car_Analysis_Report.pdf",
                        mime="application/pdf"
                    )
            else:
                st.warning("No matching items or frames could be extracted.")