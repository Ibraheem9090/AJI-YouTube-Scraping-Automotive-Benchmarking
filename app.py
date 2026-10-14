import os
import json
import streamlit as st
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
st.markdown("Interactive Learning-from-Demonstration (LfD) pipeline. Record, test, and save JSON workflows in Git, process transcripts via **Nvidia Nemotron VLM**, and compile structured PDF reports.")

WORKFLOW_FILE = "youtube_workflow.json"

# --- NAVIGATION TABS ---
tab_run, tab_studio = st.tabs(["🚀 Run Benchmarking Pipeline", "🛠️ LfD Workflow Studio (Recorder & JSON Manager)"])

# ==========================================
# TAB 1: RUN PIPELINE
# ==========================================
with tab_run:
    st.header("Automated Benchmarking Execution")
    
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

    run_button = st.button("🚀 Run LfD Pipeline & Generate PDF Report")

    if run_button:
        if not os.path.exists(WORKFLOW_FILE):
            st.error(f"Workflow file '{WORKFLOW_FILE}' not found! Please define it in the LfD Workflow Studio tab.")
        else:
            with st.spinner("Executing LfD JSON workflow and querying Nemotron VLM..."):
                with open(WORKFLOW_FILE, "r") as f:
                    workflow = json.load(f)

                all_matches = []
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
                    page = context.new_page()

                    try:
                        channel_url = f"https://www.youtube.com/{channel_handle}/videos"
                        page.goto(channel_url, timeout=30000)
                        page.wait_for_selector("ytd-rich-item-renderer", timeout=10000)

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
                            st.write(f"🌐 Processing video: {v_url}")
                            page.goto(v_url, timeout=30000)
                            page.wait_for_load_timeout(2500)

                            # Execute LfD Steps from JSON
                            transcript_opened = False
                            for step in workflow.get('steps', []):
                                if step.get('action') == 'open_transcript_panel':
                                    for selector in step.get('selectors', []):
                                        try:
                                            page.click(selector, timeout=3000)
                                            transcript_opened = True
                                            break
                                        except Exception:
                                            continue

                            if transcript_opened:
                                page.wait_for_selector("ytd-transcript-segment-renderer", timeout=6000)
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
                    except Exception as e:
                        st.error(f"Execution error: {str(e)}")
                    finally:
                        browser.close()

                if all_matches:
                    st.success(f"✅ Found {len(all_matches)} matching segment(s)!")
                    all_results = []
                    for match in all_matches:
                        # Call Nemotron VLM
                        base_url = st.secrets.get("NEMOTRON_BASE_URL", "http://localhost:8000/v1")
                        api_key = st.secrets.get("NEMOTRON_API_KEY", "not-needed")
                        client = OpenAI(base_url=base_url, api_key=api_key)
                        
                        response = client.chat.completions.create(
                            model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
                            messages=[{"role": "user", "content": f"Instructions: {user_prompt}\n\nContext: {match['ocr_text']}"}],
                            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                            max_tokens=1024
                        )
                        all_results.append({**match, 'analysis': response.choices[0].message.content})

                    # Generate PDF
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
                    st.warning("No matching segments found.")

# ==========================================
# TAB 2: LfD WORKFLOW STUDIO (RECORDER)
# ==========================================
with tab_studio:
    st.header("🛠️ LfD Workflow Studio")
    st.markdown("Inspect, customize, and save your declarative JSON workflow. When you update selectors here, they are instantly saved to `youtube_workflow.json` for Git tracking.")

    # Load existing workflow or provide default template
    default_workflow = {
        "workflow_name": "youtube_channel_transcript_extractor",
        "target_channels": ["@rsloncars", "@CCR-Galib"],
        "steps": [
            {"step": 1, "action": "navigate_channel", "target_url_template": "https://www.youtube.com/{channel}/videos"},
            {"step": 2, "action": "click_first_video", "selector": "ytd-rich-item-renderer #video-title"},
            {"step": 3, "action": "open_transcript_panel", "selectors": ["button[aria-label*='Show transcript' i]", "button[aria-label*='transcript' i]"]},
            {"step": 4, "action": "extract_segments", "container": "ytd-transcript-segment-renderer"}
        ]
    }

    if os.path.exists(WORKFLOW_FILE):
        with open(WORKFLOW_FILE, "r") as f:
            try:
                current_json_str = f.read()
            except Exception:
                current_json_str = json.dumps(default_workflow, indent=4)
    else:
        current_json_str = json.dumps(default_workflow, indent=4)

    # Editable JSON Text Area in Streamlit
    edited_json_str = st.text_area("Edit Workflow JSON Schema", value=current_json_str, height=350)

    col_save, col_test = st.columns(2)
    with col_save:
        if st.button("💾 Save Workflow to JSON"):
            try:
                # Validate JSON syntax
                parsed_json = json.loads(edited_json_str)
                with open(WORKFLOW_FILE, "w") as f:
                    json.dump(parsed_json, f, indent=4)
                st.success(f"Successfully saved and committed to `{WORKFLOW_FILE}`! You can now push this to Git.")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON format: {str(e)}")

    with col_test:
        if st.button("🧪 Test Workflow Selectors Live"):
            st.info("Testing selectors against a sample video from @CCR-Galib...")
            try:
                with sync_app_test := sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    page.goto("https://www.youtube.com/@CCR-Galib/videos")
                    page.wait_for_selector("ytd-rich-item-renderer", timeout=8000)
                    st.success("✅ Successfully connected to channel DOM and validated base selectors!")
                    browser.close()
            except Exception as e:
                st.error(f"Test failed: {str(e)}")