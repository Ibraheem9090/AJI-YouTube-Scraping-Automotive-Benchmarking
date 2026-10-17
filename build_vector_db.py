import os
import json
import re
import time
import argparse
from playwright.sync_api import sync_playwright
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound


def get_video_transcript(video_id):
    """
    Fetches transcript reliably using youtube-transcript-api with cross-version compatibility.
    """
    segments = []
    try:
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        except AttributeError:
            api = YouTubeTranscriptApi()
            transcript_list = api.fetch(video_id)

        for entry in transcript_list:
            start_sec = entry.get('start') if isinstance(entry, dict) else entry.start
            text = entry.get('text', '').strip() if isinstance(entry, dict) else entry.text.strip()
            
            if text:
                mins = int(start_sec // 60)
                secs = int(start_sec % 60)
                segments.append({
                    'start_sec': start_sec,
                    'timestamp': f"{mins}:{secs:02d}",
                    'text': text
                })
    except (TranscriptsDisabled, NoTranscriptFound):
        pass
    except Exception as e:
        print(f"    [Error] Transcript API error for {video_id}: {e}")
    return segments


def build_channel_vector_db(channel_handle="@CCR-Galib", max_videos=100):
    clean_handle = channel_handle.replace("@", "")
    output_filename = f"vector_db_{clean_handle}.json"

    print(f"🚀 Starting Channel-Wide Ingestion for {channel_handle} (Target: up to {max_videos} videos)...")
    master_transcript_index = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800}
        )
        page = context.new_page()

        channel_url = f"https://www.youtube.com/{channel_handle}/videos"
        print(f"🧭 Navigating to channel: {channel_url}")
        page.goto(channel_url, timeout=30000)

        try:
            page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            page.wait_for_timeout(4000)

        print("📜 Scrolling to hydrate the full channel video catalog...")
        for i in range(12):
            page.mouse.wheel(0, 2000)
            page.wait_for_timeout(1000)

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
        print(f"🎬 Discovered {len(video_urls)} total channel videos. Processing top {len(target_videos)} videos...")

        successful_videos = 0
        for idx, v_url in enumerate(target_videos):
            try:
                video_id = v_url.split("v=")[1].split("&")[0]
            except Exception:
                continue

            print(f"📖 Ingesting [{idx+1}/{len(target_videos)}]: {video_id}")
            
            # Fetch video title
            video_title = video_id
            try:
                page.goto(v_url, timeout=15000)
                page.wait_for_load_state("domcontentloaded", timeout=4000)
                video_title = page.title()
            except Exception:
                pass

            # Fetch transcript via API
            segments = get_video_transcript(video_id)
            if not segments:
                print(f"    ⚠️ No transcript found for {video_id}, skipping.")
                continue

            successful_videos += 1
            print(f"    ✅ Fetched {len(segments)} transcript segments.")

            # Sliding window chunking and vectorization
            for i in range(len(segments)):
                chunk_text = " ".join([segments[j]['text'] for j in range(i, min(i + 3, len(segments)))])
                words = set(re.findall(r'\w+', chunk_text.lower()))
                master_transcript_index.append({
                    'video_id': video_id,
                    'v_url': v_url,
                    'title': video_title,
                    'start_sec': segments[i]['start_sec'],
                    'timestamp': segments[i]['timestamp'],
                    'text': chunk_text.replace('\n', ' '),
                    'tokens': list(words)
                })

            time.sleep(1.0)  # Gentle delay between requests

        browser.close()

    # Save complete persistent vector database
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(master_transcript_index, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Channel-Wide Vector Database Complete!")
    print(f"📊 Indexed {len(master_transcript_index)} total chunks across {successful_videos} videos into `{output_filename}`.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube Channel-Wide Transcript Vector DB Builder")
    parser.add_argument("--channel", type=str, default="@CCR-Galib", help="YouTube Channel Handle")
    parser.add_argument("--max_videos", type=int, default=100, help="Max videos to index")
    
    args = parser.parse_args()
    build_channel_vector_db(channel_handle=args.channel, max_videos=args.max_videos)