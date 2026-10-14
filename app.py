import os
import json
import subprocess
import streamlit as st

# Function to ensure Playwright Chromium binaries are installed safely once
@st.cache_resource
def install_playwright_browsers():
    try:
        subprocess.run(["playwright", "install", "chromium"], check=True)
    except Exception as e:
        st.warning(f"Browser installation notice: {e}")

install_playwright_browsers()

from playwright.sync_api import sync_playwright
from openai import OpenAI

# ReportLab imports for PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Streamlit Page Config
st.set_page_config(page_title="AI YouTube Car Part Analyzer (LfD Studio)", layout="wide")

st.title("🚗 AI Automotive Video & Part Analyzer (LfD Studio) — By Ibraheem")
st.markdown("Declarative LfD pipeline driven by your recorded browser demonstration, processed via **Nvidia Nemotron VLM**, and compiled into structured PDF reports.")

WORKFLOW_FILE = "youtube_workflow.json"

tab_run, tab_studio = st.tabs(["🚀 Run Benchmarking Pipeline", "🛠️ LfD Workflow Studio (Recorder & JSON Manager)"])

# ==========================================
# TAB 1: RUN BENCHMARKING PIPELINE
# ==========================================
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

    run_button = st.button("🚀 Run Demonstrated LfD Workflow & Generate PDF")

    if run_button:
        if not os.path.exists(WORKFLOW_FILE):
            st.error(f"Workflow file '{WORKFLOW_FILE}' not found! Please check the Workflow Studio tab.")
        else:
            with st.spinner("Executing demonstrated LfD workflow steps via Playwright..."):
                with open(WORKFLOW_FILE, "r") as f:
                    workflow = json.load(f)

                all_matches = []
                with sync_playwright() as p:
                    # Added headless launch args required for cloud environments
                    browser = p.chromium.launch(
                        headless=True,
                        args=[
                            "--no-sandbox",
                            "--disable-setuid-sandbox",
                            "--disable-dev-shm-usage",
                            "--disable-gpu",
                            "--no-first-run",
                            "--no-zygote",
                            "--single-process"
                        ]
                    )
                    context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
                    page = context.new_page()

                    try:
                        # Step 1: Navigate to channel
                        channel_url = f"https://www.youtube.com/{channel_handle}/videos"
                        page.goto(channel_url, timeout=30000)
                        page.wait_for_selector("ytd-rich-item-renderer", timeout=10000)

                        # Step 2: Get video URLs dynamically
                        video_elements = page.locator("ytd-rich-item-renderer #video-title").all()
                        video_urls = []
                        for el in video_elements[:max_videos]:
                            href = el.get_attribute("href")
                            if href:
                                if href.startswith("/"):
                                    href = f"https://www.youtube.com{href}"
                                video_urls.append(href)

                        keyword_tokens = [t.lower() for t in search_keyword.split()]

                        for v_url in video_urls:
                            st.write(f"🌐 Navigating to video: {v_url}")
                            page.goto(v_url, timeout=30000)
                            page.wait_for_load_state("domcontentloaded")

                            # Step 3: Expand description (...more) as demonstrated in codegen
                            try:
                                page.click("button#expand, tp-yt-paper-button#expand", timeout=3000)
                            except Exception:
                                pass

                            # Step 4: Open transcript panel
                            transcript_opened = False
                            for sel in ["button[aria-label*='Show transcript' i]", "button:has-text('Show transcript')"]:
                                try:
                                    page.click(sel, timeout=3000)
                                    transcript_opened = True
                                    break
                                except Exception:
                                    continue

                            if not transcript_opened:
                                # Fallback scroll
                                page.mouse.wheel(0, 400)
                                page.wait_for_timeout(1500)
                                for sel in ["button[aria-label*='Show transcript' i]", "button:has-text('Show transcript')"]:
                                    try:
                                        page.click(sel, timeout=3000)
                                        transcript_opened = True
                                        break
                                    except Exception:
                                        continue

                            # Step 5: Extract transcript segments and match keyword
                            if transcript_opened:
                                page.wait_for_selector("ytd-transcript-segment-renderer", timeout=8000)
                                segments = page.locator("ytd-transcript-segment-renderer").all()

                                seg_texts, seg_times = [], []
                                for seg in segments:
                                    try:
                                        seg_times.append(seg.locator(".segment-timestamp").inner_text())
                                        seg_texts.append(seg.locator(".segment-text").inner_text())
                                    except Exception:
                                        continue

                                for i in range(len(seg_texts)):
                                    window_text = " ".join(seg_texts[i:min(i + 3, len(seg_texts))])
                                    if keyword_tokens and all(tok in window_text.lower() for tok in keyword_tokens):
                                        all_matches.append({
                                            'title': page.title(),
                                            'timestamp': seg_times[i],
                                            'ocr_text': window_text.replace('\n', ' ')
                                        })
                                        break
                            else:
                                st.warning("Could not open transcript panel for this video.")

                    except Exception as e:
                        st.error(f"Automation execution error: {str(e)}")
                    finally:
                        browser.close()

                if all_matches:
                    st.success(f"✅ Successfully extracted {len(all_matches)} matching segment(s) via LfD workflow!")
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

                    # Compile PDF Report
                    doc = SimpleDocTemplate("Transcript_Analysis_Report.pdf", pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
                    styles = getSampleStyleSheet()
                    story = [
                        Paragraph("🚗 Automotive Video Transcript Analysis Report — By Ibraheem", styles['Heading1']),
                        Spacer(1, 15)
                    ]
                    table_data = [[Paragraph("<b>Video</b>", styles['Normal']), Paragraph("<b>Time</b>", styles['Normal']), Paragraph("<b>Text</b>", styles['Normal']), Paragraph("<b>Analysis</b>", styles['Normal'])]]
                    for item in all_results:
                        table_data.append([Paragraph(item['title'], styles['Normal']), Paragraph(item['timestamp'], styles['Normal']), Paragraph(item['ocr_text'], styles['Normal']), Paragraph(item['analysis'], styles['Normal'])])
                    
                    t = Table(table_data, colWidths=[130, 70, 140, 200])
                    t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")), ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0"))]))
                    story.append(t)
                    doc.build(story)

                    with open("Transcript_Analysis_Report.pdf", "rb") as f:
                        st.download_button("📥 Download Structured PDF Report", f, file_name="LfD_Report.pdf", mime="application/pdf")
                else:
                    st.warning("No matching transcript segments found using the demonstrated LfD steps.")

# ==========================================
# TAB 2: LfD WORKFLOW STUDIO
# ==========================================
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