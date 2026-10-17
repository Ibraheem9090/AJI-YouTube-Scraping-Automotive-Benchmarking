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

import cv2
import yt_dlp
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

with tab_run:
    st.subheader("🔍 Search Channel Vector Index")

    channel_options = {"@CCR-Galib": "vector_db_CCR-Galib.json"}
    selected_channel = st.selectbox("Select Target Channel Index", options=list(channel_options.keys()))
    db_filename = channel_options[selected_channel]

    if os.path.exists(db_filename):
        try:
            with open(db_filename, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                st.success(f"⚡ Vector Index Active: `{db_filename}` loaded (**{len(raw_data)}** indexed chunks ready for instant RAG search).")
        except Exception as e:
            st.error(f"Error reading vector DB: {e}")
            raw_data = []
    else:
        st.error(f"Vector Database file `{db_filename}` not found. Ensure it is committed in your repo.")
        raw_data = []

    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("Target Car Feature / Keyword", value="wireless chargers")
    with col2:
        top_n_matches = st.number_input("Top Unique Vehicle Matches", min_value=1, max_value=10, value=3)

    user_prompt = st.text_area(
        "Custom AI Prompt Instructions:",
        value="Analyze this vehicle's feature specifications, design placement, and comparative implementation based on the video context."
    )

    search_button = st.button("🚀 Search Channel Benchmark")

    if search_button:
        if not raw_data:
            st.error("Vector database is empty or not loaded.")
        else:
            with st.spinner(f"Searching {len(raw_data)} indexed chunks for '{search_query}'..."):
                query_tokens = set(re.findall(r'\w+', search_query.lower()))
                scored_matches = []

                for item in raw_data:
                    chunk_tokens = set(item['tokens'])
                    common = query_tokens.intersection(chunk_tokens)
                    if common or search_query.lower().strip() in item['text'].lower():
                        score = len(common) / max(len(query_tokens), 1)
                        if search_query.lower().strip() in item['text'].lower():
                            score += 2.0
                        scored_matches.append((score, item))

                scored_matches.sort(key=lambda x: x[0], reverse=True)

                # Strict Deduplication by unique video_id
                top_matches = []
                seen_videos = set()
                for score, chunk in scored_matches:
                    vid = chunk['video_id']
                    if vid not in seen_videos:
                        seen_videos.add(vid)
                        top_matches.append(chunk)
                    if len(top_matches) >= top_n_matches:
                        break

                all_results = []
                if top_matches:
                    base_url = st.secrets.get("NEMOTRON_BASE_URL", "http://localhost:8000/v1")
                    api_key = st.secrets.get("NEMOTRON_API_KEY", "not-needed")
                    client = OpenAI(base_url=base_url, api_key=api_key)

                    for match in top_matches:
                        video_id = match['video_id']
                        exact_sec = int(match['start_sec'])
                        seek_url = f"https://www.youtube.com/watch?v={video_id}&t={exact_sec}s"
                        
                        screenshot_path = f"screenshot_{video_id}_{exact_sec}.png"
                        
                        # Extract exact frame via yt-dlp & OpenCV (No browser bot walls!)
                        success_frame = extract_exact_frame(video_id, match['start_sec'], screenshot_path)
                        if not success_frame:
                            screenshot_path = None

                        try:
                            response = client.chat.completions.create(
                                model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
                                messages=[{"role": "user", "content": f"Instructions: {user_prompt}\n\nVehicle Transcript Context: {match['text']}"}],
                                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                                max_tokens=1024
                            )
                            analysis_text = response.choices[0].message.content
                        except Exception as e:
                            analysis_text = f"Context Benchmark Analysis: {match['text']}"

                        raw_title = match['title'].replace("- YouTube", "").strip()

                        all_results.append({
                            'vehicle_title': raw_title,
                            'timestamp': match['timestamp'],
                            'seek_url': seek_url,
                            'ocr_text': match['text'],
                            'screenshot': screenshot_path if screenshot_path and os.path.exists(screenshot_path) else None,
                            'analysis': analysis_text
                        })

                st.session_state["search_results"] = all_results
                st.session_state["query_name"] = search_query

    # Display Results In-Window First
    if "search_results" in st.session_state and st.session_state["search_results"]:
        results = st.session_state["search_results"]
        q_name = st.session_state.get("query_name", "Feature Benchmark")

        st.markdown("---")
        st.subheader(f"📊 Benchmarking Results for: `{q_name}`")

        for idx, res in enumerate(results, 1):
            with st.container():
                st.markdown(f"### Vehicle #{idx}: {res['vehicle_title']}")
                
                table_data = {
                    "Field": ["Vehicle", "Timestamp Link", "Transcript Context", "AI Benchmark Analysis"],
                    "Details": [
                        res['vehicle_title'],
                        f"[{res['timestamp']}]({res['seek_url']})",
                        res['ocr_text'],
                        res['analysis']
                    ]
                }
                st.table(table_data)

                if res['screenshot'] and os.path.exists(res['screenshot']):
                    st.image(res['screenshot'], caption=f"Exact Timestamp Video Frame @ {res['timestamp']}", width=640)
                else:
                    st.info("⚠️ Frame snapshot extraction pending or unavailable for this segment.")
                
                st.markdown("---")

        # PDF Export Section
        st.subheader("📄 Export Benchmark Report")
        
        pdf_filename = f"Benchmark_Report_{q_name.replace(' ', '_')}.pdf"
        doc = SimpleDocTemplate(
            pdf_filename,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        styles = getSampleStyleSheet()
        
        cell_style = ParagraphStyle(
            'TableCell',
            parent=styles['Normal'],
            fontSize=9,
            leading=12
        )
        cell_bold = ParagraphStyle(
            'TableCellBold',
            parent=styles['Normal'],
            fontSize=9,
            leading=12,
            fontName='Helvetica-Bold'
        )

        story = [
            Paragraph(f"🚗 Automotive Benchmarking Report: {q_name}", styles['Heading1']),
            Spacer(1, 12)
        ]

        for idx, res in enumerate(results, 1):
            story.append(Paragraph(f"<b>Result #{idx}: {res['vehicle_title']}</b>", styles['Heading2']))
            story.append(Spacer(1, 6))

            t_link = f'<a href="{res["seek_url"]}" color="#2563EB"><u>{res["timestamp"]}</u></a>'
            
            table_content = [
                [Paragraph("Vehicle", cell_bold), Paragraph(res['vehicle_title'], cell_style)],
                [Paragraph("Timestamp", cell_bold), Paragraph(t_link, cell_style)],
                [Paragraph("Context", cell_bold), Paragraph(res['ocr_text'], cell_style)],
                [Paragraph("AI Analysis", cell_bold), Paragraph(res['analysis'], cell_style)],
            ]

            t = Table(table_content, colWidths=[100, 440])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            
            story.append(t)
            story.append(Spacer(1, 10))

            if res['screenshot'] and os.path.exists(res['screenshot']):
                story.append(RLImage(res['screenshot'], width=440, height=247))
                story.append(Spacer(1, 20))

        doc.build(story)

        with open(pdf_filename, "rb") as pdf_file:
            st.download_button(
                label="📥 Download Structured Comparative PDF Report",
                data=pdf_file,
                file_name=pdf_filename,
                mime="application/pdf"
            )

with tab_studio:
    st.header("🛠️ LfD Workflow Studio (Read-Only Schema)")
    if os.path.exists(WORKFLOW_FILE):
        with open(WORKFLOW_FILE, "r") as fn:
            current_json_str = fn.read()
    else:
        current_json_str = "{}"
    st.text_area("Workflow JSON Schema", value=current_json_str, height=350, disabled=True)