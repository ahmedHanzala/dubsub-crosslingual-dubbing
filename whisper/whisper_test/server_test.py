import socket
from whisper_online import *
import numpy as np
import librosa  
import io
from voice_activity_controller import VoiceActivityController
import soundfile

SAMPLING_RATE = 16000
model = "large-v3"
src_lan = "ar"
tgt_lan = "en"
use_vad_result = True
min_sample_length = 1 * SAMPLING_RATE
max_sample_length = 5*SAMPLING_RATE

complete_text = ''
transcription = ''
count = 0
final_processing_pending = False
out = []
out_len = 0

asr = FasterWhisperASR(src_lan, model)
asr.set_translate_task()
tokenizer = create_tokenizer(tgt_lan)
#online = OnlineASRProcessor(asr, tokenizer)
vad = VoiceActivityController(use_vad_result=use_vad_result)


HOST = 'localhost'
PORT = 4048

def audio_stream(conn):
    buffer = b''  # Buffer to store received data
    while True:
        raw_bytes = conn.recv(4096)  # Use a larger buffer size
        if not raw_bytes:
            break
        buffer += raw_bytes
        while len(buffer) >= 2048:
            yield buffer[:2048]  # Send chunks of 1280 bytes
            buffer = buffer[2048:]

    # After the loop, if there's any remaining data in the buffer, yield it
    if buffer:
        yield buffer

        
#pseudocode
"""
1. recieve audio buffer
2. Transcribe via faster whisper (complete_text = complete_text + transcription obtained)
3. Add buffer to buffer history
4. if buffer exceeds 20s flush buffer history
5. Repeat 1-4



"""

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        audio_generator = audio_stream(conn)
        for iter in vad.detect_user_speech(audio_generator):
            raw_bytes=  iter[0]
            is_final =  iter[1]
            if  raw_bytes:
                sf = soundfile.SoundFile(io.BytesIO(raw_bytes), channels=1,endian="LITTLE",samplerate=SAMPLING_RATE, subtype="PCM_16",format="RAW")
                audio, _ = librosa.load(sf,sr=SAMPLING_RATE)
                out.append(audio)
                out_len += len(audio)

            
            if (is_final or out_len >= min_sample_length) and out_len>0:
                a = np.concatenate(out)
                print("LEN OF BUFFER:", len(a))
                transcription = asr.transcribe(a, init_prompt=complete_text)
                transcription = transcription[0].text
                print('transcription: ', transcription)
                #conn.sendall(transcription.encode('utf-8'))

                
            if out_len > max_sample_length:
                out = []
                out_len = 0
                complete_text = complete_text + transcription
                count = 1
            
            if count == 0:
                conn.sendall(transcription.encode('utf-8'))
            else:
                x = complete_text + transcription
                conn.sendall(x.encode('utf-8'))
        


            
"""
1. send transcription when not exceeded limit
2. if limit exceeds save the current transcription

"""