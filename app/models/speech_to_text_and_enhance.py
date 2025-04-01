import os
import json
import requests
from flask import current_app
from transformers import pipeline
import speech_recognition as sr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plaidml.keras
# plaidml.keras.install_backend()


# Load pre-trained pipelines for speech-to-text conversion (ASR).
# Here we use Hugging Face's Wav2Vec2 model.
asr_pipeline = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

# Load the custom-trained text summarization model.
summarization_model = load_model("app/models/saved_model/summarization_model.h5")

def load_tokenizer(tokenizer_path: str):
    """
    Load a Keras Tokenizer from a JSON file.
    
    Args:
        tokenizer_path (str): Path to the tokenizer JSON file.
        
    Returns:
        A Keras Tokenizer instance.
    """
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)
    return tokenizer

# Load the tokenizer (assumes it was saved during training).
tokenizer = load_tokenizer("app/models/saved_model/tokenizer.json")

def transcribe_audio(file_path: str) -> str:
    """
    Convert an audio file to text using the Hugging Face ASR pipeline.
    
    Args:
        file_path (str): Path to the audio file.
        
    Returns:
        str: The transcribed text.
    """
    try:
        result = asr_pipeline(file_path)
        return result.get("text", "")
    except Exception as e:
        current_app.logger.error("Error in ASR: %s", e)
        return "Speech-to-text processing failed."

def save_transcript(transcript: str, output_path: str) -> None:
    """
    Save the transcript to a text file.
    
    Args:
        transcript (str): The transcribed text.
        output_path (str): Path to the output file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(transcript)

def generate_summary(text: str, max_length: int = 20) -> str:
    """
    Generate a summary for the provided text using the custom-trained summarization model.
    
    Args:
        text (str): The text to summarize.
        max_length (int): Maximum sequence length used during training.
        
    Returns:
        str: The generated summary as a string.
    """
    # Convert input text to a sequence.
    sequence = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(sequence, maxlen=max_length, padding='post')
    
    # Predict output using the summarization model.
    predictions = summarization_model.predict(padded_seq)
    # For each time step, choose the index with highest probability.
    predicted_indices = predictions.argmax(axis=-1)[0]
    
    # Convert indices back to words.
    summary_words = []
    for idx in predicted_indices:
        if idx == 0:  # Skip padding index.
            continue
        word = tokenizer.index_word.get(idx, '')
        if word:
            summary_words.append(word)
    
    summary = " ".join(summary_words)
    return summary

def verify_subject_and_extract_keywords(transcript: str) -> str:
    """
    Verify the subject matter of the transcript using the Bing Web Search API,
    and extract keywords to append to the transcript. These keywords help ensure
    the transcript is contextually accurate.
    
    Environment Variables:
        - BING_API_KEY: Your Bing Web Search API key.
        - BING_ENDPOINT: (Optional) The Bing search endpoint (default: "https://api.bing.microsoft.com/v7.0/search")
    
    Args:
        transcript (str): The text transcript.
    
    Returns:
        str: The transcript with appended verified keywords.
    """
    api_key = os.environ.get("BING_API_KEY")
    endpoint = os.environ.get("BING_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search")
    
    if not api_key:
        raise ValueError("BING_API_KEY is not set in environment variables.")
    
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {
        "q": transcript,
        "textDecorations": True,
        "textFormat": "HTML",
        "count": 5
    }
    
    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()
        
        keywords = []
        web_pages = search_results.get("webPages", {}).get("value", [])
        for item in web_pages:
            title = item.get("name", "")
            for word in title.split():
                if len(word) > 3 and word.lower() not in keywords:
                    keywords.append(word.lower())
        
        enhanced_transcript = (
            transcript +
            "\n\n[Verified Keywords: " +
            ", ".join(keywords) +
            "]"
        )
        return enhanced_transcript
    except Exception as e:
        current_app.logger.error("Error in subject verification: %s", e)
        return transcript + "\n\n[Keyword extraction failed.]"

def process_audio_file(file_path: str, transcript_output_path: str) -> dict:
    """
    Process an audio file by converting it to text, summarizing the text,
    and verifying the subject matter to extract keywords. The transcript
    is saved to a file for future training or fine-tuning.
    
    Args:
        file_path (str): Path to the input audio file.
        transcript_output_path (str): Path to save the generated transcript.
    
    Returns:
        dict: Contains 'enhanced_transcript' (transcript with verified keywords)
              and 'summary' (the generated summary).
    """
    # Convert speech to text.
    transcript = transcribe_audio(file_path)
    # Save the raw transcript.
    save_transcript(transcript, transcript_output_path)
    
    # Generate a summary using the custom-trained model.
    summary = generate_summary(transcript, max_length=20)
    
    # Verify subject matter and extract keywords.
    enhanced_transcript = verify_subject_and_extract_keywords(transcript)
    
    return {"enhanced_transcript": enhanced_transcript, "summary": summary}

if __name__ == "__main__":
    # Example usage:
    # Ensure you have an audio file from the LibriSpeech train-clean-360 dataset.
    audio_file_path = "app/models/data/audio/example.wav"  # Replace with an actual audio file path.
    transcript_output_path = "app/models/data/text/example_transcript.txt"
    
    results = process_audio_file(audio_file_path, transcript_output_path)
    print("Enhanced Transcript:")
    print(results["enhanced_transcript"])
    print("\nSummary:")
    print(results["summary"])
