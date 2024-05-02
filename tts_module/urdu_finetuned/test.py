import json
from pathlib import Path
from app import gen_voice, tts, update_speakers, languages

def generate_voices_from_file(file_path):
    with open(file_path, 'r') as f:
        texts = json.load(f)

    speakers = update_speakers()

    for text in texts:
        for speaker in speakers:
            gen_voice(text, speaker, speed=0.8, english="English")

if __name__ == "__main__":
    generate_voices_from_file(Path('texts.json'))