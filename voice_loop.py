import soundfile as sf
from kokoro import KPipeline
import os, sys, queue, threading, tempfile, subprocess, numpy as np
import sounddevice as sd, soundfile as sf
from faster_whisper import WhisperModel
from dotenv import load_dotenv
from openai import OpenAI

# --- Load config ---
load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WHISPER_SIZE = os.getenv("WHISPER_MODEL", "base")  # "tiny" for faster

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Audio config ---
SAMPLE_RATE = 16000
CHANNELS = 1

# Load Whisper model once
print(f"Loading Whisper model: {WHISPER_SIZE}")
model = WhisperModel(WHISPER_SIZE, compute_type="int8")  # efficient for CPU

KOKORO_VOICE = os.getenv("KOKORO_SPEAKER", "af_heart")  # try: af_heart, af_sky, af_alloy, af_nicole, am_michael
KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", "1.2"))  # 1.0 is normal, 1.2 is 20% faster
kokoro_pipeline = KPipeline(lang_code="a") 

def speak(text: str):
    if not text:
        return
    try:
        # Kokoro returns a generator of (grapheme_seq, phoneme_seq, audio_np)
        gen = kokoro_pipeline(text, voice=KOKORO_VOICE, speed=KOKORO_SPEED)
        # Stitch chunks and save a 24 kHz WAV
        audio_chunks = []
        for _, _, audio in gen:
            audio_chunks.append(audio)
        if not audio_chunks:
            raise RuntimeError("Kokoro produced no audio.")
        audio = np.concatenate(audio_chunks)
        wav_path = tempfile.mktemp(suffix=".wav")
        sf.write(wav_path, audio, 24000)
        subprocess.run(["afplay", wav_path])
    except Exception as e:
        print(f"[TTS fallback] Kokoro failed: {e}. Using 'say'.")
        subprocess.run(["say", text])

def record_wav() -> str:
    """Record audio until Enter is pressed again."""
    print("Press Enter to start recording...")
    input()
    print("🎙️ Recording... Press Enter again to stop.")

    stop_flag = {"stop": False}
    def wait_for_enter():
        input()
        stop_flag["stop"] = True

    stopper = threading.Thread(target=wait_for_enter, daemon=True)
    stopper.start()

    q = queue.Queue()
    def cb(indata, frames, time_, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    frames = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype="float32", callback=cb):
        while not stop_flag["stop"]:
            try:
                frames.append(q.get(timeout=0.1))
            except queue.Empty:
                pass

    if not frames:
        return ""

    data = np.concatenate(frames, axis=0)
    path = tempfile.mktemp(suffix=".wav")
    sf.write(path, data, SAMPLE_RATE)
    print(f"✅ Saved recording to {path}")
    if len(data) < SAMPLE_RATE * 0.3:  # less than ~0.3s
        return ""
    return path

def transcribe(path: str) -> str:
    """Transcribe WAV file with Whisper."""
    if not path: return ""
    segments, info = model.transcribe(path, language="en", vad_filter=True)
    text = " ".join([seg.text for seg in segments]).strip()
    return text

# Global conversation history
conversation_history = [
    {"role": "system", "content": "You are a friendly, conversational voice assistant optimized for text-to-speech. CRITICAL: Write exactly as you would speak to someone in person. NEVER use numbered lists, bullet points, or formatting symbols. Instead, use natural speech patterns like 'first', 'second', 'third', 'next', 'finally', 'also', 'additionally'. Convert all technical content into conversational speech. For example, instead of '1. Insert: O(log n)', say 'First, let's talk about insertion. This typically takes logarithmic time on average.' Avoid reading out any symbols, numbers, or formatting - just speak the content naturally and conversationally."}
]

def clean_text_for_tts(text: str) -> str:
    """Clean text to be more TTS-friendly."""
    import re
    
    # Remove numbered lists and convert to natural speech
    text = re.sub(r'\d+\.\s*', '', text)  # Remove "1. ", "2. " etc.
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove bold formatting
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Remove italic formatting
    text = re.sub(r'`([^`]+)`', r'\1', text)  # Remove code formatting
    
    # Replace technical notation with spoken equivalents
    text = re.sub(r'O\(([^)]+)\)', r'big O of \1', text)  # O(log n) -> big O of log n
    text = re.sub(r'O\(', 'big O of ', text)  # Handle incomplete O() notation
    
    # Clean up multiple spaces and newlines
    text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    
    return text.strip()

def chat_llm(user_text: str) -> str:
    """Send text to LLM and get response with conversation history."""
    if not user_text: return ""
    
    # Add user message to history
    conversation_history.append({"role": "user", "content": user_text})
    
    # Limit history to last 5 exchanges (10 messages + system message) to prevent token overflow
    if len(conversation_history) > 11:  # system + 5 exchanges (10 messages)
        # Keep system message and last 5 exchanges
        conversation_history[:] = [conversation_history[0]] + conversation_history[-10:]
    
    resp = client.chat.completions.create(model=OPENAI_MODEL, messages=conversation_history)
    assistant_response = resp.choices[0].message.content.strip()
    
    # Clean the response for TTS
    assistant_response = clean_text_for_tts(assistant_response)
    
    # Add assistant response to history
    conversation_history.append({"role": "assistant", "content": assistant_response})
    
    return assistant_response

def main():
    print("🎤 Local Voice Assistant ready. Ctrl+C to quit.")
    while True:
        try:
            wav = record_wav()
            if not wav:
                print("⚠️ No audio captured.")
                continue
            user_text = transcribe(wav)
            if not user_text:
                print("🤔 Didn’t catch that.")
                speak("I didn't catch that.")
                continue
            print(f"👤 You: {user_text}")
            reply = chat_llm(user_text)
            print(f"🤖 Assistant: {reply}")
            speak(reply)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()
