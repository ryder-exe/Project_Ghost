from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
import os
import sys

app = Flask(__name__)
CORS(app)

try:
    # Ensure that a backend framework is available for the transformers library
    summarizer = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')
except Exception as e:
    summarizer = None
    print("Error initializing the summarizer pipeline. Ensure PyTorch or TensorFlow is installed.")
    print(e)
    sys.exit(1)  # Exit if a critical dependency is missing

@app.route('/')
def index():
    return "Hello, Flask is running!"

@app.route('/test', methods=['POST'])
def test():
    global summarizer
    if not summarizer:
        return jsonify({'error': 'Summarizer pipeline is not initialized. Ensure PyTorch or TensorFlow is installed.'}), 500

    data = request.get_json()
    youtube_video = data.get('ytlink')
    if not youtube_video or "=" not in youtube_video:
        return jsonify({'error': 'Invalid YouTube link provided.'}), 400

    try:
        video_id = youtube_video.split("=")[1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        return jsonify({'error': f'Error fetching transcript: {str(e)}'}), 500

    result = " ".join([i['text'] for i in transcript])

    # Split the transcript into chunks for summarization
    num_iters = len(result) // 1000 + 1
    summarized_text = []

    try:
        for i in range(num_iters):
            start = i * 1000
            end = (i + 1) * 1000
            chunk = result[start:end]
            out = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
            summarized_text.append(out[0]['summary_text'])
    except Exception as e:
        return jsonify({'error': f'Error during summarization: {str(e)}'}), 500

    return jsonify({'summarized_text': summarized_text})

if __name__ == "__main__":
    app.run(debug=True)
