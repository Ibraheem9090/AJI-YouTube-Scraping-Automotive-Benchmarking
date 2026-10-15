import os
import json
import subprocess
import base64
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

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

st.set_page_config(page_title="AI YouTube Car Part Analyzer (LfD Studio)", layout="wide")

st.title("🚗 AI Automotive Video & Part Analyzer (LfD Studio) — By Ibraheem")
st.markdown("Declarative LfD pipeline driven by your recorded browser demonstration, processed via **Nvidia Nemotron VLM**, and compiled into structured PDF reports with video screenshots.")

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

    run_button = st.button("🚀 Run Demonstrated LfD Workflow & Generate PDF")

    if run_button:
        if not os.path.exists(WORKFLOW_FILE):
            st.error(f"Workflow file '{WORKFLOW_FILE}' not found! Please check the Workflow Studio tab.")
        else:
            status_box = st.status("Initializing LfD Workflow...", expanded=True)
            
            with status_box:
                all_matches = []
                with sync_playwright() as p:
                    try:
                        st.write("🌐 Launching headless Chromium container...")
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

                        # Dismiss cookie banner if present
                        try:
                            consent_btn = page.locator("button[aria-label*='Accept all' i], ytd-button-renderer:has-text('Accept all')").first
                            if consent_btn.is_visible(timeout=3000):
                                consent_btn.click()
                                st.write("🍪 Dismissed YouTube cookie consent banner.")
                        except Exception:
                            pass

                        # Scroll down to hydrate video elements
                        st.write("📜 Scrolling page to trigger lazy-loaded video grid...")
                        for _ in range(3):
                            page.mouse.wheel(0, 800)
                            page.wait_for_timeout(1500)

                        # Bulletproof JS extraction
                        st.write("🔎 Extracting video URLs via robust DOM JavaScript query...")
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
                            st.write(f"🌐 Opening video page: {v_url}")
                            page.goto(v_url, timeout=30000)
                            page.wait_for_load_state("domcontentloaded")
                            page.wait_for_timeout(3000)

                            # Expand description (...more)
                            st.write("🔍 Expanding video description (...more)...")
                            expanded = False
                            for sel in ["button#expand", "tp-yt-paper-button#expand", "text=...more"]:
                                try:
                                    page.click(sel, timeout=3000)
                                    expanded = True
                                    st.write(f"✔️ Clicked description expander using: `{sel}`")
                                    break
                                except Exception:
                                    continue
                            
                            if not expanded:
                                page.mouse.wheel(0, 300)
                                page.wait_for_timeout(1000)

                            # Open transcript panel
                            st.write("📂 Looking for 'Show transcript' button...")
                            transcript_opened = False
                            for sel in [
                                "button[aria-label*='Show transcript' i]", 
                                "button:has-text('Show transcript')", 
                                "text=Show transcript",
                                "ytd-button-renderer:has-text('Transcript')"
                            ]:
                                try:
                                    page.click(sel, timeout=4000)
                                    transcript_opened = True
                                    st.write(f"✔️ Opened transcript panel using selector: `{sel}`")
                                    break
                                except Exception:
                                    continue

                            if not transcript_opened:
                                try:
                                    page.evaluate("""() => {
                                        const buttons = Array.from(document.querySelectorAll('button, yt-formatted-string'));
                                        const btn = buttons.find(b => b.innerText && b.innerText.toLowerCase().includes('transcript'));
                                        if (btn) btn.click();
                                    }""")
                                    page.wait_for_timeout(2000)
                                    transcript_opened = True
                                    st.write("✔️ JS fallback click executed for transcript.")
                                except Exception as e:
                                    st.write(f"❌ Failed JS fallback: {e}")

                            # Extract segments from active YouTube engagement panel
                            if transcript_opened:
                                st.write("⏳ Waiting for transcript API segments to load...")
                                page.wait_for_timeout(5000)

                                st.write("🔎 Extracting transcript segments from active YouTube engagement panel...")
                                segment_data = page.evaluate("""() => {
                                    const panel = document.querySelector('ytd-engagement-panel-section-list-renderer[visibility="VISIBLE"]') || document;
                                    
                                    function querySelectorAllDeep(selector, root = panel) {
                                        let results = Array.from(root.querySelectorAll(selector));
                                        const allElements = root.querySelectorAll('*');
                                        for (const el of allElements) {
                                            if (el.shadowRoot) {
                                                results.push(...querySelectorAllDeep(selector, el.shadowRoot));
                                            }
                                        }
                                        return results;
                                    }

                                    let segments = querySelectorAllDeep('ytd-transcript-segment-renderer');
                                    if (segments.length === 0) {
                                        segments = querySelectorAllDeep('.segment-rendered, [class*="transcript-segment"], yt-formatted-string.segment-text');
                                    }
                                    
                                    return segments.map((seg, index) => {
                                        const timeEl = seg.querySelector('.segment-timestamp, [class*="timestamp"]') || 
                                                       (seg.shadowRoot ? seg.shadowRoot.querySelector('.segment-timestamp') : null);
                                        const textEl = seg.querySelector('.segment-text, [class*="text"]') || 
                                                       (seg.shadowRoot ? seg.shadowRoot.querySelector('.segment-text') : null);
                                        
                                        return {
                                            index: index,
                                            timestamp: timeEl ? timeEl.innerText.trim() : "0:00",
                                            text: textEl ? textEl.innerText.trim() : seg.innerText.trim()
                                        };
                                    });
                                }""")

                                st.write(f"📊 Extracted {len(segment_data)} transcript segment(s) from panel.")

                                for i in range(len(segment_data)):
                                    window_text = " ".join([segment_data[j]['text'] for j in range(i, min(i + 3, len(segment_data)))])
                                    window_lower = window_text.lower()
                                    if keyword_tokens and all(tok in window_lower for tok in keyword_tokens):
                                        st.write(f"✨ Match found at timestamp {segment_data[i]['timestamp']}! Clicking timestamp segment...")
                                        
                                        # Click the timestamp row to seek the video player to this timestamp
                                        try:
                                            page.evaluate(f"""(idx) => {{
                                                const panel = document.querySelector('ytd-engagement-panel-section-list-renderer[visibility="VISIBLE"]') || document;
                                                const segments = panel.querySelectorAll('ytd-transcript-segment-renderer');
                                                if (segments[idx]) {{
                                                    segments[idx].click();
                                                }}
                                            }}""", i)
                                            page.wait_for_timeout(2500) # Allow video player to render the frame
                                        except Exception as click_err:
                                            st.write(f"Note on timestamp click: {click_err}")

                                        # Capture video element screenshot
                                        screenshot_path = f"screenshot_match_{len(all_matches)}.png"
                                        try:
                                            player = page.locator("video, #movie_player").first
                                            player.screenshot(path=screenshot_path)
                                            st.write(f"📸 Captured video frame screenshot at {segment_data[i]['timestamp']}!")
                                        except Exception:
                                            # Fallback: capture full browser page screenshot
                                            page.screenshot(path=screenshot_path)
                                            st.write(f"📸 Captured page screenshot at {segment_data[i]['timestamp']}!")

                                        all_matches.append({
                                            'title': page.title(),
                                            'timestamp': segment_data[i]['timestamp'],
                                            'ocr_text': window_text.replace('\n', ' '),
                                            'screenshot': screenshot_path
                                        })
                                        break
                            else:
                                st.write("❌ Could not open transcript panel.")

                    except Exception as e:
                        st.error(f"Automation execution error: {str(e)}")
                    finally:
                        try:
                            browser.close()
                        except Exception:
                            pass

                if all_matches:
                    status_box.update(label="✅ Workflow Execution Complete & Successful!", state="complete", expanded=False)
                    st.success(f"Successfully processed {len(all_matches)} matching segment(s)!")
                    
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
                    st.warning("Check the live status box above to verify video URL extraction.")

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