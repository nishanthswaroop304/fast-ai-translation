#!/usr/bin/env python3
import os
import json
import re
import sys
from groq import Groq  # Ensure groq is installed and configured correctly

def transcribe_audio(filename, assumed_duration=90.0):
    client = Groq()
    with open(filename, "rb") as file:
        translation = client.audio.translations.create(
            file=(filename, file.read()),
            model="whisper-large-v3",  # Using the multilingual model
            response_format="verbose_json",
            temperature=0.0
        )
    # Use model_dump() instead of dict() to avoid deprecation warnings.
    translation_dict = translation.model_dump()
    segments = translation_dict.get("segments")
    if segments and isinstance(segments, list) and len(segments) > 0:
        pass  # Use the returned timestamped segments.
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
        translation_dict["segments"] = segments
    return translation_dict

def call_chat_translation(text):
    """
    Calls the Groq chat endpoint to translate the provided text to English.
    The assistant is instructed to strictly return only the English translation.
    If unable to translate, it should return the original text.
    """
    client = Groq()
    messages = [
        {"role": "system", "content": "You are a translation assistant. Your task is to translate the provided text strictly into English and output only the translated text. If you are unable to translate, return the original text as is."},
        {"role": "user", "content": f"Translate the following text to English: {text}"}
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
    if len(sys.argv) < 2:
        print("Usage: python translation_test.py <audio_file_path>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found!")
        sys.exit(1)
    
    print(f"Processing transcription for '{audio_file}'...\n")
    result = transcribe_audio(audio_file)
    segments = result.get("segments", [])
    
    if not segments:
        print("No segments found.")
        sys.exit(1)
    
    print("Translated Text for All Segments:")
    for i, seg in enumerate(segments):
        original_text = seg.get("text", "")
        translated_text = call_chat_translation(original_text)
        print(f"Segment {i+1}: {translated_text}")

if __name__ == '__main__':
    main()
