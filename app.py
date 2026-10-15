import os
import json
import subprocess
import re
import time
import streamlit as st

@st.cache_resource
def ensure_packages():
    try:
        subprocess.run(["playwright", "install", "chromium"], check=True)
    except Exception as e:
        st.warning(f"Browser binary setup note: {e}")

ensure_packages()

from playwright.sync_api import sync_playwright
from openai import OpenAI
import yt_dlp

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

st.set_page_config(page_title="AI YouTube Car Part Analyzer", layout="wide")

st.title("🚗 AI Automotive Video & Part Analyzer — By Ibraheem")
st.markdown("Automated multi-video transcript benchmarking pipeline powered by **yt-dlp VTT subtitle parsing**, processed via **Nvidia Nemotron VLM**, and compiled into structured PDF reports with hyperlinked timestamps and video screenshots.")

WORKFLOW_FILE = "youtube_workflow.json"

tab_run, tab_studio = st.tabs(["🚀 Run Benchmarking Pipeline", "🛠️ LfD Workflow Studio (Schema Inspector)"])

def parse_vtt_file(filepath):
    segments = []
    if not os.path.exists(filepath):
        return segments
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        blocks = content.split('\n\n')
        for block in blocks:
            lines = block.strip().split('\n')
            time_line_idx = -1
            for idx, line in enumerate(lines):
                if '-->' in line:
                    time_line_idx = idx
                    break
            
            if time_line_idx != -1:
                time_line = lines[time_line_idx]
                parts = time_line.split('-->')
                start_str = parts[0].strip().split()[0]
                
                t_parts = start_str.replace(',', '.').split(':')
                start_sec = 0.0
                if len(t_parts) == 3:
                    start_sec = float(t_parts[0]) * 3600 + float(t_parts[1]) * 60 + float(t_parts[2])
                elif len(t_parts) == 2:
                    start_sec = float(t_parts[0]) * 60 + float(t_parts[1])
                
                text_lines = lines[time_line_idx + 1:]
                text = " ".join(text_lines).strip()
                text = re.sub(r'<[^>]+>', '', text)
                text = re.sub(r'\{[^}]+\}', '', text)
                
                if text:
                    mins = int(start_sec // 60)
                    secs = int(start_sec % 60)
                    segments.append({
                        'start_sec': start_sec,
                        'timestamp': f"{mins}:{secs:02d}",
                        'text': text
                    })
    except Exception as e:
        print(f"VTT parse error: {e}")
        
    return segments

with tab_run:
    st.header("Automated Benchmarking Execution")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        channel_options = {"@rsloncars": "@rsloncars", "@CCR-Galib": "@CCR-Galib"}
        selected_channel_name = st.selectbox("Select YouTube Channel", options=list(channel_options.keys()))
        channel_handle = channel_options[selected_channel_name]
        
        search_keyword = st.text_input("Target Keyword / Item", value="wireless chargers")

    with col2:
        max_videos = st.slider("Max Videos to Process", min_value=1, max_value=5, value=3)
        user_prompt = st.text_area(
            "Custom AI Prompt Instructions:",
            value="Analyze the component's specifications, features, and context mentioned in this transcript segment."
        )

    run_button = st.button("🚀 Run Benchmarking Pipeline & Generate PDF")

    if run_button:
        status_box = st.status("Initializing Benchmarking Pipeline...", expanded=True)
        
        with status_box:
            all_matches = []
            
            with sync_playwright() as p:
                try:
                    st.write("🌐 Launching headless Chromium container to discover videos...")
                    browser = p.chromium.launch(
                        headless=True,
                        args=[
                            "--no-sandbox",
                            "--disable-setuid-sandbox",
                            "--disable-dev-shm-usage",
                            "--disable-accelerated-2d-canvas",
                            "--no-first-run",
                            "--no-zygote",
                            "--single-process",
                            "--disable-gpu"
                        ]
                    )
                    context = browser.new_context(
                        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                        viewport={"width": 1280, "height": 800}
                    )
                    page = context.new_page()

                    channel_url = f"https://www.youtube.com/{channel_handle}/videos"
                    st.write(f"🧭 Navigating to channel: {channel_url}")
                    page.goto(channel_url, timeout=30000)
                    
                    try:
                        page.wait_for_load_state("networkidle", timeout=10000)
                    except Exception:
                        page.wait_for_timeout(4000)

                    # Scroll down to hydrate video grid
                    st.write("📜 Scrolling page to trigger lazy-loaded video grid...")
                    for _ in range(3):
                        page.mouse.wheel(0, 800)
                        page.wait_for_timeout(1500)

                    video_urls = page.evaluate("""() => {
                        const anchors = Array.from(document.querySelectorAll('a'));
                        const urls = [];
                        anchors.forEach(a => {
                            if (a.href && a.href.includes('/watch?v=') && !urls.includes(a.href)) {
                                if (!a.href.includes('/shorts/') && !a.href.includes('/playlist?')) {
                                    urls.push(a.href.split('&')[0]);
                                }
                            }
                        });
                        return [...new Set(urls)];
                    }""")

                    target_videos = video_urls[:max_videos]
                    st.write(f"🎬 Found {len(video_urls)} total videos. Processing top {len(target_videos)} video(s): {target_videos}")
                    
                    search_lower = search_keyword.lower().strip()
                    keyword_tokens = [t.lower() for t in search_keyword.split()]

                    for idx, v_url in enumerate(target_videos):
                        try:
                            video_id = v_url.split("v=")[1].split("&")[0]
                        except Exception:
                            continue

                        # Polite delay between requests to avoid HTTP 429 rate limits
                        if idx > 0:
                            time.sleep(3)

                        st.write(f"📖 Downloading auto-subtitles via yt-dlp for video ID: {video_id}")
                        sub_file_prefix = f"sub_{video_id}"
                        vtt_filename = f"{sub_file_prefix}.en.vtt"
                        
                        ydl_opts = {
                            'writeautomaticsub': True,
                            'subtitleslangs': ['en'],
                            'subtitlesformat': 'vtt',
                            'skip_download': True,
                            'outtmpl': sub_file_prefix,
                            'quiet': True,
                            'no_warnings': True,
                        }
                        
                        try:
                            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                                ydl.download([v_url])
                        except Exception as ydl_err:
                            st.warning(f"yt-dlp download note for {video_id}: {ydl_err}")

                        segments = parse_vtt_file(vtt_filename)
                        
                        if os.path.exists(vtt_filename):
                            try:
                                os.remove(vtt_filename)
                            except Exception:
                                pass

                        if not segments:
                            st.warning(f"Could not retrieve transcript segments for {video_id}.")
                            continue

                        st.write(f"📊 Retrieved {len(segments)} transcript segments for {video_id}. Scanning for keyword matches...")

                        # Flexible matching: checks phrase substring or presence of main tokens
                        matched_indices = []
                        i = 0
                        while i < len(segments):
                            window_text = " ".join([segments[j]['text'] for j in range(i, min(i + 3, len(segments)))])
                            window_lower = window_text.lower()
                            
                            is_match = False
                            if search_lower in window_lower:
                                is_match = True
                            elif keyword_tokens and all(tok in window_lower for tok in keyword_tokens):
                                is_match = True
                            elif keyword_tokens and any(tok in window_lower for tok in keyword_tokens) and len(keyword_tokens) == 1:
                                is_match = True

                            if is_match:
                                matched_indices.append(i)
                                i += 6
                            else:
                                i += 1

                        st.write(f"✨ Found {len(matched_indices)} matching segment(s) in video {video_id}.")

                        for m_idx in matched_indices:
                            match_seg = segments[m_idx]
                            seek_url = f"https://www.youtube.com/watch?v={video_id}&t={int(match_seg['start_sec'])}s"
                            st.write(f"🌐 Opening video player at timestamp {match_seg['timestamp']}: {seek_url}")
                            
                            page.goto(seek_url, timeout=30000)
                            page.wait_for_load_state("domcontentloaded")
                            page.wait_for_timeout(4000)

                            try:
                                page.locator("button[aria-label*='Accept all' i]").click(timeout=2000)
                            except Exception:
                                pass

                            screenshot_path = f"screenshot_{video_id}_{int(match_seg['start_sec'])}.png"
                            try:
                                page.screenshot(path=screenshot_path)
                                st.write(f"📸 Captured screenshot for timestamp {match_seg['timestamp']}!")
                            except Exception as ss_err:
                                st.write(f"Screenshot error: {ss_err}")

                            window_text = " ".join([segments[j]['text'] for j in range(m_idx, min(m_idx + 3, len(segments)))])
                            all_matches.append({
                                'title': page.title(),
                                'timestamp': match_seg['timestamp'],
                                'seek_url': seek_url,
                                'ocr_text': window_text.replace('\n', ' '),
                                'screenshot': screenshot_path
                            })

                except Exception as e:
                    st.error(f"Automation execution error: {str(e)}")
                finally:
                    try:
                        browser.close()
                    except Exception:
                        pass

            if all_matches:
                status_box.update(label="✅ Workflow Execution Complete & Successful!", state="complete", expanded=False)
                st.success(f"Successfully processed {len(all_matches)} matching segment(s) across videos with screenshots and hyperlinks!")
                
                all_results = []
                base_url = st.secrets.get("NEMOTRON_BASE_URL", "http://localhost:8000/v1")
                api_key = st.secrets.get("NEMOTRON_API_KEY", "not-needed")
                client = OpenAI(base_url=base_url, api_key=api_key)

                for match in all_matches:
                    try:
                        response = client.chat.completions.create(
                            model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
                            messages=[{"role": "user", "content": f"Instructions: {user_prompt}\n\nContext: {match['ocr_text']}"}],
                            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                            max_tokens=1024
                        )
                        analysis_text = response.choices[0].message.content
                    except Exception as e:
                        analysis_text = f"Nemotron VLM Analysis Error: {str(e)}"

                    all_results.append({**match, 'analysis': analysis_text})

                doc = SimpleDocTemplate("Transcript_Analysis_Report.pdf", pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
                styles = getSampleStyleSheet()
                story = [
                    Paragraph("🚗 Automotive Video Transcript Analysis Report — By Ibraheem", styles['Heading1']),
                    Spacer(1, 15)
                ]
                
                for idx, item in enumerate(all_results, 1):
                    story.append(Paragraph(f"<b>Match {idx}: {item['title']}</b>", styles['Heading2']))
                    
                    timestamp_link = f'<a href="{item["seek_url"]}" color="#2563EB"><u>{item["timestamp"]}</u></a>'
                    story.append(Paragraph(f"<b>Timestamp:</b> {timestamp_link}", styles['Normal']))
                    
                    story.append(Paragraph(f"<b>Context:</b> {item['ocr_text']}", styles['Normal']))
                    story.append(Spacer(1, 8))
                    
                    if os.path.exists(item['screenshot']):
                        story.append(RLImage(item['screenshot'], width=380, height=215))
                        story.append(Spacer(1, 10))

                    story.append(Paragraph("<b>AI Analysis:</b>", styles['Heading3']))
                    story.append(Paragraph(item['analysis'], styles['Normal']))
                    story.append(Spacer(1, 20))

                doc.build(story)

                with open("Transcript_Analysis_Report.pdf", "rb") as f:
                    st.download_button("📥 Download Structured PDF Report (Hyperlinked & Screenshots)", f, file_name="LfD_Report.pdf", mime="application/pdf")
            else:
                status_box.update(label="⚠️ Execution finished, but no matching segments were captured.", state="error", expanded=True)
                st.warning("Try a different search keyword or check channel video availability.")

with tab_studio:
    st.header("🛠️ LfD Workflow Studio (Read-Only Schema)")
    st.markdown("Inspect your recorded LfD workflow JSON schema stored in Git.")

    if os.path.exists(WORKFLOW_FILE):
        with open(WORKFLOW_FILE, "r") as fn:
            current_json_str = fn.read()
    else:
        current_json_str = "{}"

    st.text_area("Workflow JSON Schema", value=current_json_str, height=350, disabled=True)