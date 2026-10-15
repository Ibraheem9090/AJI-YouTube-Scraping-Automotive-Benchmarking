import os
import json
import subprocess
import streamlit as st

@st.cache_resource
def ensure_playwright_installed():
    try:
        subprocess.run(["playwright", "install", "chromium"], check=True)
    except Exception as e:
        st.warning(f"Browser binary setup note: {e}")

ensure_playwright_installed()

from playwright.sync_api import sync_playwright
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

st.set_page_config(page_title="AI YouTube Car Part Analyzer (LfD Studio)", layout="wide")

st.title("🚗 AI Automotive Video & Part Analyzer (LfD Studio) — By Ibraheem")
st.markdown("Declarative LfD pipeline driven by robust transcript extraction, processed via **Nvidia Nemotron VLM**, and compiled into structured PDF reports with video screenshots.")

WORKFLOW_FILE = "youtube_workflow.json"

tab_run, tab_studio = st.tabs(["🚀 Run Benchmarking Pipeline", "🛠️ LfD Workflow Studio (Recorder & JSON Manager)"])

with tab_run:
    st.header("Automated LfD Benchmarking Execution")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        channel_options = {"@rsloncars": "@rsloncars", "@CCR-Galib": "@CCR-Galib"}
        selected_channel_name = st.selectbox("Select YouTube Channel", options=list(channel_options.keys()))
        channel_handle = channel_options[selected_channel_name]
        
        search_keyword = st.text_input("Target Keyword / Item", value="wireless chargers")

    with col2:
        max_videos = st.slider("Max Videos to Process", min_value=1, max_value=5, value=1)
        user_prompt = st.text_area(
            "Custom AI Prompt Instructions:",
            value="Analyze the component's specifications, features, and context mentioned in this transcript segment."
        )

    run_button = st.button("🚀 Run Benchmarking Pipeline & Generate PDF")

    if run_button:
        if not os.path.exists(WORKFLOW_FILE):
            st.error(f"Workflow file '{WORKFLOW_FILE}' not found! Please check the Workflow Studio tab.")
        else:
            status_box = st.status("Initializing Benchmarking Pipeline...", expanded=True)
            
            with status_box:
                all_matches = []
                
                # 1. Fetch Video URLs via Playwright
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

                        st.write(f"🎬 Found {len(video_urls)} video URL(s) to process: {video_urls[:max_videos]}")
                        
                        keyword_tokens = [t.lower() for t in search_keyword.split()]

                        for v_url in video_urls[:max_videos]:
                            # Extract video ID (e.g. https://www.youtube.com/watch?v=XYZ -> XYZ)
                            try:
                                video_id = v_url.split("v=")[1].split("&")[0]
                            except Exception:
                                continue

                            st.write(f"📖 Fetching transcript via API for video ID: {video_id}")
                            try:
                                raw_transcript = YouTubeTranscriptApi.get_transcript(video_id)
                            except Exception as api_err:
                                st.warning(f"Could not fetch transcript for {video_id}: {api_err}")
                                continue

                            # Format transcript segments
                            segments = []
                            for entry in raw_transcript:
                                start_sec = entry['start']
                                minutes = int(start_sec // 60)
                                seconds = int(start_sec % 60)
                                timestamp_str = f"{minutes}:{seconds:02d}"
                                segments.append({
                                    'start_sec': start_sec,
                                    'timestamp': timestamp_str,
                                    'text': entry['text']
                                })

                            st.write(f"📊 Retrieved {len(segments)} transcript entries. Searching for keyword tokens: {keyword_tokens}")

                            matched_index = -1
                            for i in range(len(segments)):
                                window_text = " ".join([segments[j]['text'] for j in range(i, min(i + 3, len(segments)))])
                                window_lower = window_text.lower()
                                if keyword_tokens and all(tok in window_lower for tok in keyword_tokens):
                                    matched_index = i
                                    break

                            if matched_index != -1:
                                match_seg = segments[matched_index]
                                st.write(f"✨ Match found at timestamp {match_seg['timestamp']} ({match_seg['start_sec']}s)!")
                                
                                # Open video at exact timestamp using URL parameter &t=
                                seek_url = f"{v_url}&t={int(match_seg['start_sec'])}s"
                                st.write(f"🌐 Opening video player at timestamp: {seek_url}")
                                page.goto(seek_url, timeout=30000)
                                page.wait_for_load_state("domcontentloaded")
                                page.wait_for_timeout(3500) # Wait for player to seek and render frame

                                # Dismiss cookie/consent dialogs if blocking view
                                try:
                                    page.locator("button[aria-label*='Accept all' i]").click(timeout=2000)
                                except Exception:
                                    pass

                                # Capture screenshot of the video player frame
                                screenshot_path = f"screenshot_{video_id}.png"
                                try:
                                    player = page.locator("video, #movie_player").first
                                    player.screenshot(path=screenshot_path)
                                    st.write("📸 Captured video player frame screenshot successfully!")
                                except Exception:
                                    page.screenshot(path=screenshot_path)
                                    st.write("📸 Captured page screenshot successfully!")

                                window_text = " ".join([segments[j]['text'] for j in range(matched_index, min(matched_index + 3, len(segments)))])
                                all_matches.append({
                                    'title': page.title(),
                                    'timestamp': match_seg['timestamp'],
                                    'ocr_text': window_text.replace('\n', ' '),
                                    'screenshot': screenshot_path
                                })
                            else:
                                st.write("⚠️ Keyword not found in this video's transcript.")

                    except Exception as e:
                        st.error(f"Automation execution error: {str(e)}")
                    finally:
                        try:
                            browser.close()
                        except Exception:
                            pass

                if all_matches:
                    status_box.update(label="✅ Workflow Execution Complete & Successful!", state="complete", expanded=False)
                    st.success(f"Successfully processed {len(all_matches)} matching segment(s) with screenshots!")
                    
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

                    # Compile PDF Report with Embedded Screenshots
                    doc = SimpleDocTemplate("Transcript_Analysis_Report.pdf", pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
                    styles = getSampleStyleSheet()
                    story = [
                        Paragraph("🚗 Automotive Video Transcript Analysis Report — By Ibraheem", styles['Heading1']),
                        Spacer(1, 15)
                    ]
                    
                    for idx, item in enumerate(all_results, 1):
                        story.append(Paragraph(f"<b>Match {idx}: {item['title']}</b>", styles['Heading2']))
                        story.append(Paragraph(f"<b>Timestamp:</b> {item['timestamp']}", styles['Normal']))
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
                        st.download_button("📥 Download Structured PDF Report (With Screenshots)", f, file_name="LfD_Report.pdf", mime="application/pdf")
                else:
                    status_box.update(label="⚠️ Execution finished, but no matching segments were captured.", state="error", expanded=True)
                    st.warning("Try a different search keyword or check channel video availability.")

with tab_studio:
    st.header("🛠️ LfD Workflow Studio")
    st.markdown("Inspect and manage your recorded LfD workflow JSON schema stored in Git.")

    if os.path.exists(WORKFLOW_FILE):
        with open(WORKFLOW_FILE, "r") as f:
            current_json_str = f.read()
    else:
        current_json_str = "{}"

    edited_json_str = st.text_area("Edit Workflow JSON Schema", value=current_json_str, height=350)

    if st.button("💾 Save Workflow to JSON"):
        try:
            parsed_json = json.loads(edited_json_str)
            with open(WORKFLOW_FILE, "w") as f:
                json.dump(parsed_json, f, indent=4)
            st.success(f"Successfully saved to `{WORKFLOW_FILE}`!")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON format: {str(e)}")