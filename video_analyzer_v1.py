import os
import cv2
import base64
import time
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

if not openai_api_key and not groq_api_key:
    st.error("Please set at least one API key (OPENAI_API_KEY or GROQ_API_KEY) in the environment variables.")
    st.stop()

# Initialize AI clients
openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None

# Streamlit UI
st.title("AI Inference Demo")

# Video selection
video_option = st.radio("Do you want to upload a video or use the demo video?", ("Use Demo Video", "Upload Your Own Video"))

if video_option == "Upload Your Own Video":
    uploaded_file = st.file_uploader("Upload your video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
    else:
        st.stop()
else:
    video_path = "temp_video.mp4"
    if not os.path.exists(video_path):
        st.error("Demo video file not found. Please upload a video.")
        st.stop()

# AI model selection
ai_choice = st.selectbox("Choose AI Model:", ["Slow Inference (OpenAI GPT-4o-mini)", "Fast Inference (Groq Llama-3.2-11b-vision-preview)"])

if st.button("Run AI Analysis"):
    st.write("### AI Analysis Output:")
    common_prompt = "Describe the scene in exactly 10 words or fewer. Avoid extra details. Focus only on pedestrians, number of vehicles, traffic signals and its colors"
    
    avg_response_time_placeholder = st.empty()
    cols = st.columns([3, 2])  # Video on left, analysis on right
    video_placeholder = cols[0].empty()
    analysis_container = cols[1].container()
    
    analysis_placeholder = analysis_container.empty()
    response_times = []
    analysis_results = {}
    
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(i * total_frames / 10) for i in range(10)]  # Select 10 frames evenly
    
    frame_index = 0
    processed_count = 0
    
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        
        # Rotate frame 90 degrees to the right
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        # Display current frame
        video_placeholder.image(frame, channels="BGR")
        
        if frame_index in frame_indices and processed_count < 10:
            # Encode and send frame to AI model
            _, buffer = cv2.imencode(".jpg", frame)
            base64_image = base64.b64encode(buffer).decode("utf-8")
            
            start_time = time.time()
            
            if ai_choice == "Slow Inference (OpenAI GPT-4o-mini)" and openai_client:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "user", "content": [
                            {"type": "text", "text": common_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ]},
                    ],
                )
                content = response.choices[0].message.content
            
            elif ai_choice == "Fast Inference (Groq Llama-3.2-11b-vision-preview)" and groq_client:
                response = groq_client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",
                    messages=[
                        {"role": "user", "content": [
                            {"type": "text", "text": common_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ]},
                    ],
                )
                content = response.choices[0].message.content
            else:
                st.error("Invalid AI selection or missing API key.")
                break
            
            end_time = time.time()
            response_time = end_time - start_time
            response_times.append(response_time)
            
            analysis_results[frame_index] = f"Frame {processed_count + 1}: {content} (Response Time: {response_time:.2f}s)"
            processed_count += 1
        
        # Show analysis for the current frame if it was processed
        if frame_index in analysis_results:
            analysis_placeholder.markdown(analysis_results[frame_index])
        
        frame_index += 1
    
    video.release()
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    avg_response_time_placeholder.markdown(f"#### Average Processing Time Per Frame: {avg_response_time:.2f} seconds")
