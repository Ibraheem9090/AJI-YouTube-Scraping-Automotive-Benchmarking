import os
import urllib.parse
import base64
import streamlit as st
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from openai import OpenAI

# ReportLab imports for PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Streamlit Page Config
st.set_page_config(page_title="AI YouTube Car Part Analyzer (Transcript-Only)", layout="wide")

st.title("🚗 AI Automotive Video & Part Analyzer (Transcript-Only Pipeline) — By Ibraheem")
st.markdown("Automated multi-modal pipeline for Chinese car review channels (`@rsloncars`, `@CCR-Galib`), utilizing direct transcript timestamp extraction, processed via **Nvidia Nemotron-3 Nano Omni Reasoning VLM**, and compiled into a structured PDF report.")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Pipeline Configuration")

channel_options = {
    "@rsloncars": "https://www.youtube.com/@rsloncars",
    "@CCR-Galib": "https://www.youtube.com/@CCR-Galib"
}

selected_channel_name = st.sidebar.selectbox(
    "Select YouTube Channel",
    options=list(channel_options.keys())
)

channel_or_video_url = channel_options[selected_channel_name]

search_keyword = st.sidebar.text_input(
    "Target Keyword / Item (e.g., screen, battery, dashboard)", 
    value="wireless chargers"
)

max_videos = st.sidebar.slider("Max Videos to Process", min_value=1, max_value=5, value=1)

st.sidebar.markdown("---")

user_prompt = st.sidebar.text_area(
    "Custom AI Prompt Instructions:",
    value="Analyze the component's specifications, features, and context mentioned in this transcript segment.",
    help="Custom prompt passed to the VLM reasoning module along with transcript text."
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
    if "@" in url and not url.endswith(("/videos", "/shorts", "/streams")):
        url = url.rstrip("/") + "/videos"
        
    ydl_opts = {
        'extract_flat': 'in_playlist',
        'skip_download': True,
        'playlistend': limit
    }
    videos = []
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if 'entries' in info:
                for entry in info['entries']:
                    vid_id = entry.get('id')
                    if vid_id and not vid_id.startswith('UC'):
                        videos.append({
                            'id': vid_id,
                            'title': entry.get('title', 'Unknown Title'),
                            'url': f"https://www.youtube.com/watch?v={vid_id}"
                        })
            else:
                vid_id = info.get('id')
                if vid_id and not vid_id.startswith('UC'):
                    videos.append({
                        'id': vid_id,
                        'title': info.get('title', 'Unknown Title'),
                        'url': url
                    })
    except Exception as e:
        st.error(f"Error fetching channel metadata: {str(e)}")
    return videos

def phase1_transcript_check(video_id, keyword):
    """Phase 1: Robust transcript fetching using list_transcripts with safe fallback handling."""
    transcript = None
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to find English or Chinese transcript tracks
        target_track = None
        for t in transcript_list:
            if any(lang in t.language_code.lower() for lang in ['en', 'zh']):
                target_track = t
                break
        
        # If specific language not found, take the first available track
        if not target_track:
            for t in transcript_list:
                target_track = t
                break
                
        if target_track:
            transcript = target_track.fetch()
            
    except Exception as e:
        st.warning(f"Could not retrieve transcript list for video ID {video_id}: {str(e)}")
        return None

    if not transcript:
        return None

    matches = []
    try:
        keyword_tokens = [token.lower() for token in keyword.split()]
        
        # Sliding window across consecutive subtitle fragments to catch multi-word and pluralized phrases
        for i in range(len(transcript)):
            window_text = " ".join([transcript[j]['text'] for j in range(i, min(i + 3, len(transcript)))])
            window_lower = window_text.lower()
            
            if keyword_tokens and all(token in window_lower for token in keyword_tokens):
                matches.append({
                    'start_time': transcript[i]['start'],
                    'text': window_text.replace('\n', ' ')
                })
    except Exception as e:
        st.warning(f"Transcript parsing warning for video {video_id}: {str(e)}")
        return None
        
    return matches if matches else None

def analyze_with_nemotron_vlm(user_prompt, context_text):
    """Sends transcript context to nvidia/nemotron-3-nano-omni-30b-a3b-reasoning with thinking mode enabled."""
    try:
        base_url = st.secrets.get("NEMOTRON_BASE_URL", "http://localhost:8000/v1")
        api_key = st.secrets.get("NEMOTRON_API_KEY", "not-needed")
        
        client = OpenAI(base_url=base_url, api_key=api_key)
        
        messages_content = [
            {
                "type": "text", 
                "text": f"Instructions: {user_prompt}\n\nTranscript Segment / Context: {context_text}"
            }
        ]

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

def generate_pdf_report(results_data, filename="Transcript_Analysis_Report.pdf"):
    """Compiles final structured PDF table report using ReportLab."""
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
    
    story.append(Paragraph("🚗 Automotive Video Transcript Analysis Report — By Ibraheem", title_style))
    story.append(Paragraph("Generated via Transcript-Only Nemotron VLM Pipeline", styles['Normal']))
    story.append(Spacer(1, 15))

    table_data = [[
        Paragraph("<b>Video / Channel</b>", styles['Normal']),
        Paragraph("<b>Timestamp</b>", styles['Normal']),
        Paragraph("<b>Transcript Text</b>", styles['Normal']),
        Paragraph("<b>Nemotron Reasoning & Analysis</b>", styles['Normal'])
    ]]

    for item in results_data:
        meta_text = f"<b>Title:</b> {item['title']}"
        detail_text = f"<b>AI Analysis:</b> {item['analysis']}"

        table_data.append([
            Paragraph(meta_text, styles['Normal']),
            Paragraph(str(item['timestamp']), styles['Normal']),
            Paragraph(item['ocr_text'], styles['Normal']),
            Paragraph(detail_text, styles['Normal'])
        ])

    col_widths = [130, 70, 140, 200]
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
    with st.spinner("Fetching transcripts and querying Nemotron-3 Nano Omni VLM..."):
        videos = get_videos_from_source(channel_or_video_url, limit=max_videos)
        
        if not videos:
            st.warning("No valid videos found from the selected URL.")
        else:
            all_results = []
            
            for vid in videos:
                st.write(f"Processing video: **{vid['title']}**")
                video_id = extract_video_id(vid['url'])
                
                transcript_matches = phase1_transcript_check(video_id, search_keyword) if video_id else None
                
                if transcript_matches:
                    st.success(f"✅ Transcript found! Located {len(transcript_matches)} keyword match(es).")
                    for match in transcript_matches[:3]:
                        mins = int(match['start_time'] // 60)
                        secs = int(match['start_time'] % 60)
                        timestamp_str = f"{mins:02d}:{secs:02d}"
                        
                        ai_analysis = analyze_with_nemotron_vlm(
                            user_prompt=user_prompt, 
                            context_text=match['text']
                        )
                        
                        all_results.append({
                            'title': vid['title'],
                            'timestamp': timestamp_str,
                            'ocr_text': match['text'],
                            'analysis': ai_analysis
                        })
                else:
                    st.warning(f"ℹ️ Transcript missing or keyword '{search_keyword}' not found for video: {vid['title']}.")

            if all_results:
                pdf_filename = generate_pdf_report(all_results)
                st.success("🎉 Analysis complete and structured PDF report successfully generated!")
                
                with open(pdf_filename, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Download Structured PDF Report",
                        data=pdf_file,
                        file_name="Transcript_Car_Analysis_Report.pdf",
                        mime="application/pdf"
                    )
            else:
                st.warning("No matching transcript segments could be extracted across the selected videos.")