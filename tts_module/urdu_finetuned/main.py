import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices


tts = TextToSpeech(ar_checkpoint= "./models/urdu_tts.pth")

#text = "آپ کيا کررہے ہو؟"

voices = ['urdu-all']

preset = "fast"

# This will generate the audio.
gens = [tts.tts_with_preset(voice=voice, preset=preset) for voice in voices]



def generate_audio(text, voice, preset="fast", sample_rate=16000):
    voice_samples, conditioning_latents = load_voice(voice)
    gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                          preset=preset)
    
    audio_file = f"{voice}_audio.wav"
    torchaudio.save(audio_file, gen.squeeze(0).cpu(), sample_rate)
    
    return audio_file

