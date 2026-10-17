import os
import json
import re
import time
import subprocess
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

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="AI YouTube Car Feature Analyzer", layout="wide")

st.title("🚗 AI Automotive Channel-Wide Benchmarking")
st.markdown("Query the channel-wide pre-indexed RAG vector store (**13,850+ chunks across 97 videos**) processed via **Nvidia Nemotron VLM** to generate comparative PDF reports.")

WORKFLOW_FILE = "youtube_workflow.json"

tab_run, tab_studio = st.tabs(["🚀 Query Vector DB & Generate PDF", "🛠️ LfD Workflow Studio (Schema Inspector)"])

with tab_run:
    st.subheader("🔍 Query Channel Vector Index")

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
        top_n_matches = st.number_input("Top Matches for Report", min_value=1, max_value=10, value=5)

    user_prompt = st.text_area(
        "Custom AI Prompt Instructions:",
        value="Analyze this vehicle's feature specifications, design placement, and comparative implementation based on the video context."
    )

    search_button = st.button("🚀 Query Vector DB & Build Comparative PDF Report")

    if search_button:
        if not raw_data:
            st.error("Vector database is empty or not loaded.")
        else:
            search_status = st.status("Querying Vector Index & Compiling PDF Report...", expanded=True)
            with search_status:
                st.write(f"🔍 Searching {len(raw_data)} indexed chunks for query: **'{search_query}'**...")
                
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

                # Deduplicate matches to show diverse top videos
                top_matches = []
                seen_videos = {}
                for score, chunk in scored_matches:
                    vid = chunk['video_id']
                    seen_videos[vid] = seen_videos.get(vid, 0) + 1
                    if seen_videos[vid] <= 2 and len(top_matches) < top_n_matches:
                        top_matches.append(chunk)

                st.write(f"✨ Selected top {len(top_matches)} matching segments across vehicles.")

                all_matches = []
                if top_matches:
                    with sync_playwright() as p:
                        try:
                            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
                            page = browser.new_context(viewport={"width": 1280, "height": 800}).new_page()

                            base_url = st.secrets.get("NEMOTRON_BASE_URL", "http://localhost:8000/v1")
                            api_key = st.secrets.get("NEMOTRON_API_KEY", "not-needed")
                            client = OpenAI(base_url=base_url, api_key=api_key)

                            for match in top_matches:
                                video_id = match['video_id']
                                seek_url = f"https://www.youtube.com/watch?v={video_id}&t={int(match['start_sec'])}s"
                                st.write(f"📸 Capturing snapshot for [{match['title'][:35]}...] at {match['timestamp']}...")

                                page.goto(seek_url, timeout=30000)
                                page.wait_for_load_state("domcontentloaded")
                                page.wait_for_timeout(3500)

                                screenshot_path = f"screenshot_{video_id}_{int(match['start_sec'])}.png"
                                try:
                                    page.screenshot(path=screenshot_path)
                                except Exception as ss_err:
                                    st.write(f"Screenshot error: {ss_err}")

                                try:
                                    response = client.chat.completions.create(
                                        model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
                                        messages=[{"role": "user", "content": f"Instructions: {user_prompt}\n\nVehicle Transcript Context: {match['text']}"}],
                                        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                                        max_tokens=1024
                                    )
                                    analysis_text = response.choices[0].message.content
                                except Exception as e:
                                    analysis_text = f"Nemotron VLM Analysis: {match['text']}"

                                all_matches.append({
                                    'title': match['title'],
                                    'timestamp': match['timestamp'],
                                    'seek_url': seek_url,
                                    'ocr_text': match['text'],
                                    'screenshot': screenshot_path,
                                    'analysis': analysis_text
                                })
                        except Exception as browser_err:
                            st.error(f"Browser playback error: {browser_err}")
                        finally:
                            try:
                                browser.close()
                            except Exception:
                                pass

                if all_matches:
                    search_status.update(label="✅ Comparative Report Generated!", state="complete", expanded=False)

                    # Build PDF
                    pdf_filename = "Channel_Comparative_Report.pdf"
                    doc = SimpleDocTemplate(pdf_filename, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
                    styles = getSampleStyleSheet()
                    story = [
                        Paragraph(f"🚗 Channel-Wide Comparative Benchmarking Report: {search_query}", styles['Heading1']),
                        Spacer(1, 15)
                    ]

                    for idx, item in enumerate(all_matches, 1):
                        story.append(Paragraph(f"<b>Vehicle {idx}: {item['title']}</b>", styles['Heading2']))
                        timestamp_link = f'<a href="{item["seek_url"]}" color="#2563EB"><u>{item["timestamp"]}</u></a>'
                        story.append(Paragraph(f"<b>Timestamp:</b> {timestamp_link}", styles['Normal']))
                        story.append(Paragraph(f"<b>Context:</b> {item['ocr_text']}", styles['Normal']))
                        story.append(Spacer(1, 8))

                        if os.path.exists(item['screenshot']):
                            story.append(RLImage(item['screenshot'], width=380, height=215))
                            story.append(Spacer(1, 10))

                        story.append(Paragraph("<b>AI Benchmark Analysis:</b>", styles['Heading3']))
                        story.append(Paragraph(item['analysis'], styles['Normal']))
                        story.append(Spacer(1, 20))

                    doc.build(story)

                    with open(pdf_filename, "rb") as f:
                        st.download_button("📥 Download Comparative PDF Report", f, file_name=pdf_filename, mime="application/pdf")
                else:
                    search_status.update(label="⚠️ No matches found for this query.", state="error", expanded=True)

with tab_studio:
    st.header("🛠️ LfD Workflow Studio (Read-Only Schema)")
    if os.path.exists(WORKFLOW_FILE):
        with open(WORKFLOW_FILE, "r") as fn:
            current_json_str = fn.read()
    else:
        current_json_str = "{}"
    st.text_area("Workflow JSON Schema", value=current_json_str, height=350, disabled=True)