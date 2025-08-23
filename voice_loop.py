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

def chat_llm(user_text: str) -> str:
    """Send text to LLM and get response."""
    if not user_text: return ""
    messages = [
        {"role": "system", "content": "You are a friendly, conversational voice assistant. Respond naturally as if speaking to someone in person. Use conversational language, natural pauses (indicated by commas), and clear pronunciation. Keep responses concise but warm and engaging. Avoid overly technical jargon unless specifically asked. Use contractions like 'you're', 'I'm', 'that's' to sound more natural when spoken aloud."},
        {"role": "user", "content": user_text}
    ]
    resp = client.chat.completions.create(model=OPENAI_MODEL, messages=messages)
    return resp.choices[0].message.content.strip()

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
