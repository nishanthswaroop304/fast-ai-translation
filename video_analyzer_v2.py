import os
import json
import re
import streamlit as st
import yt_dlp
from groq import Groq  # Ensure groq is installed and configured correctly

# Helper function to extract the YouTube video ID.
def extract_video_id(url):
    if "youtube.com" in url:
        try:
            return url.split("v=")[-1].split("&")[0]
        except IndexError:
            return ""
    elif "youtu.be" in url:
        return url.split("/")[-1]
    else:
        return ""

# Download audio from the YouTube video using a unique filename based on the video ID.
def download_audio(youtube_url):
    video_id = extract_video_id(youtube_url)
    output_path = f"audio_{video_id}.mp3"
    ydl_opts = {
        'format': 'bestaudio/best',
        'extractaudio': True,
        'audioformat': 'mp3',
        'outtmpl': output_path,
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return output_path

# Transcribe audio using Groq’s Whisper model.
# If no timestamped segments are returned, split the full text into sentences with estimated timings.
def transcribe_audio(filename, assumed_duration=90.0):
    client = Groq()
    with open(filename, "rb") as file:
        translation = client.audio.translations.create(
            file=(filename, file.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
            temperature=0.0
        )
    translation_dict = translation.dict()
    segments = translation_dict.get("segments")
    if segments and isinstance(segments, list) and len(segments) > 0:
        pass  # Use the timestamped segments returned.
    else:
        full_text = translation_dict.get("text", "").strip()
        sentences = re.split(r'(?<=[.!?])\s+', full_text)
        num_sentences = len(sentences)
        if num_sentences == 0:
            segments = []
        else:
            segment_duration = assumed_duration / num_sentences
            segments = []
            for i, sentence in enumerate(sentences):
                segments.append({
                    "start": round(i * segment_duration, 2),
                    "end": round((i + 1) * segment_duration, 2),
                    "text": sentence
                })
    return {"text": translation_dict.get("text", ""), "segments": segments}

def main():
    st.title("Fast AI Inference -- Real Time Translation")
    st.write("Select a video option:")
    option = st.radio("Video Option", ("Stock Video", "Custom URL"))
    
    # Set the URL based on the selected option.
    if option == "Stock Video":
        youtube_url = "https://www.youtube.com/watch?v=abFz6JgOMCk&list=PLs7zUO7VPyJ5DV1iBRgSw2uDl832n0bLg&index=1"
        st.write("Using stock video.")
    else:
        youtube_url = st.text_input("Enter YouTube Video URL:")
        if not youtube_url:
            st.info("Please enter a YouTube video URL.")
    
    # Always show the Prepare Captions button.
    if st.button("Prepare Captions"):
        if option == "Custom URL" and (not youtube_url or not youtube_url.startswith("http")):
            st.error("Invalid URL. Please enter a valid YouTube URL.")
            return

        with st.spinner("Processing transcription..."):
            audio_file = download_audio(youtube_url)
            result = transcribe_audio(audio_file)
        segments = result["segments"]
        st.success("Video is ready for real-time translation!")
        
        segments_json = json.dumps(segments)
        delay_offset = 1  # 1-second delay offset for each segment.
        video_id = extract_video_id(youtube_url)

        html_code = f"""
        <html>
          <head>
            <script>
              // Load the IFrame Player API code asynchronously.
              var tag = document.createElement('script');
              tag.src = "https://www.youtube.com/iframe_api";
              var firstScriptTag = document.getElementsByTagName('script')[0];
              firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);

              var player;
              // The transcription segments passed from Python.
              var segments = {segments_json};

              function onYouTubeIframeAPIReady() {{
                player = new YT.Player('player', {{
                  height: '360',
                  width: '640',
                  videoId: '{video_id}',
                  events: {{
                    'onReady': onPlayerReady
                  }}
                }});
              }}

              function onPlayerReady(event) {{
                setInterval(checkCaption, 500);
              }}

              function checkCaption() {{
                if (player && player.getCurrentTime) {{
                  var currentTime = player.getCurrentTime();
                  var captionText = "";
                  for (var i = 0; i < segments.length; i++) {{
                    var seg = segments[i];
                    if (currentTime >= (seg.start + {delay_offset}) && currentTime <= (seg.end + {delay_offset})) {{
                      captionText = seg.text;
                      break;
                    }}
                  }}
                  document.getElementById("captions").innerHTML = captionText;
                }}
              }}
            </script>
          </head>
          <body style="background-color: #121212;">
            <div id="player"></div>
            <div id="captions" style="color: white; font-size:20px; margin-top:10px; font-weight:bold;"></div>
          </body>
        </html>
        """
        st.components.v1.html(html_code, height=500, scrolling=False)

if __name__ == "__main__":
    main()
