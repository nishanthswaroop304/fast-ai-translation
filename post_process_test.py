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
    # Remove any existing file to force a fresh download.
    if os.path.exists(output_path):
        os.remove(output_path)
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
            model="whisper-large-v3",  # Using the multilingual model
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

# Call Groq's chat endpoint to translate text to English.
def call_chat_translation(text):
    client = Groq()
    messages = [
        {
            "role": "system", 
            "content": "You are a translation assistant. Your task is to translate the provided text strictly into English and output only the translated text. If you are unable to translate, return the original text as is."
        },
        {
            "role": "user", 
            "content": f"Translate the following text to English: {text}"
        }
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        max_completion_tokens=1024,
        top_p=1,
        stop=None,
        stream=False
    )
    return chat_completion.choices[0].message.content.strip()

def main():
    st.title("Fast AI Inference -- Real Time Language Translation")
    
    # Move the radio button outside the form so it updates dynamically.
    video_option = st.radio("Select Video Option", options=["Stock Video", "Custom URL"], key="video_option")
    
    with st.form(key="url_form"):
        if video_option == "Custom URL":
            input_youtube_url = st.text_input("Enter YouTube Video URL:", key="input_youtube_url")
        else:
            input_youtube_url = "https://www.youtube.com/watch?v=abFz6JgOMCk&list=PLs7zUO7VPyJ5DV1iBRgSw2uDl832n0bLg&index=1"
            st.info("Using Stock Video.")
        submit_url = st.form_submit_button("Prepare Audio")
    
    # Process the URL and prepare audio if the form is submitted.
    if submit_url:
        if not input_youtube_url.startswith("http"):
            st.error("Invalid URL. Please enter a valid YouTube URL.")
            return
        
        st.session_state["user_youtube_url"] = input_youtube_url

        with st.spinner("Extracting audio from the video..."):
            audio_file = download_audio(input_youtube_url)
        st.session_state["audio_file"] = audio_file
        st.success("Audio extraction successful!")
        
        with st.spinner("Transcribing segments..."):
            result = transcribe_audio(audio_file)
            segments = result.get("segments", [])
            translated_segments = []
            for seg in segments:
                original_text = seg.get("text", "")
                translated_text = call_chat_translation(original_text)
                seg["text"] = translated_text
                translated_segments.append(seg)
            st.session_state["translated_segments"] = translated_segments
        st.success("Transcription complete!")
    
    # Show the Play and Translate button once audio and segments are ready.
    if st.session_state.get("audio_file") is not None and st.session_state.get("translated_segments") is not None:
        if st.button("Play and Translate"):
            segments_json = json.dumps(st.session_state["translated_segments"])
            video_id = extract_video_id(st.session_state["user_youtube_url"])
            html_code = f"""
            <html>
              <head>
                <script>
                  // Load the YouTube IFrame API code asynchronously.
                  var tag = document.createElement('script');
                  tag.src = "https://www.youtube.com/iframe_api";
                  var firstScriptTag = document.getElementsByTagName('script')[0];
                  firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);

                  var player;
                  // The translated transcription segments passed from Python.
                  var segments = {segments_json};

                  function onYouTubeIframeAPIReady() {{
                    player = new YT.Player('player', {{
                      height: '360',
                      width: '640',
                      videoId: '{video_id}',
                      playerVars: {{
                        autoplay: 1,
                        controls: 0,
                        modestbranding: 1,
                        rel: 0
                      }},
                      events: {{
                        'onReady': onPlayerReady
                      }}
                    }});
                  }}

                  function onPlayerReady(event) {{
                    // Automatically start the video.
                    event.target.playVideo();
                    setInterval(checkCaption, 500);
                  }}

                  function checkCaption() {{
                    if (player && player.getCurrentTime) {{
                      var currentTime = player.getCurrentTime();
                      var captionText = "";
                      // Iterate over the segments and display the matching translation.
                      for (var i = 0; i < segments.length; i++) {{
                        var seg = segments[i];
                        if (currentTime >= seg.start && currentTime <= seg.end) {{
                          captionText = seg.text;
                          break;
                        }}
                      }}
                      document.getElementById("captions").innerHTML = captionText;
                    }}
                  }}
                </script>
              </head>
              <body style="background-color: #121212; color: white; text-align: center;">
                <div id="player"></div>
                <div id="captions" style="font-size:20px; margin-top:10px; font-weight:bold;"></div>
              </body>
            </html>
            """
            st.components.v1.html(html_code, height=500, scrolling=False)

# Initialize session state keys before calling main().
if "user_youtube_url" not in st.session_state:
    st.session_state["user_youtube_url"] = ""
if "audio_file" not in st.session_state:
    st.session_state["audio_file"] = None
if "translated_segments" not in st.session_state:
    st.session_state["translated_segments"] = None

if __name__ == "__main__":
    main()
