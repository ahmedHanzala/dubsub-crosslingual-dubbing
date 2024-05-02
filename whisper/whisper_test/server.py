import socket
from whisper_online import *
import numpy as np
import librosa  
import io
from voice_activity_controller import VoiceActivityController
import soundfile
import time

SAMPLING_RATE = 16000
model = "large-v3"
src_lan = "ar"
tgt_lan = "en"
use_vad_result = True
min_sample_length = 0.25 * SAMPLING_RATE #default 1*sr
sentence_end = 5*SAMPLING_RATE 

complete_text = ''
final_processing_pending = False
out = []
out_len = 0

asr = FasterWhisperASR(src_lan, model)
asr.set_translate_task()
tokenizer = create_tokenizer(tgt_lan)
online = OnlineASRProcessor(asr, tokenizer)
vad = VoiceActivityController(use_vad_result=use_vad_result)

HOST = 'localhost'
PORT = 5003

def audio_stream(conn):
    buffer = b''  # Buffer to store received data
    while True:
        raw_bytes = conn.recv(1024)  # Use a larger buffer size 4096 default
        if not raw_bytes:
            break
        buffer += raw_bytes
        while len(buffer) >= 1024:
            yield buffer[:1024]  # Send chunks of 2048 default
            buffer = buffer[1024:]

    # After the loop, if there's any remaining data in the buffer, yield it
    if buffer:
        yield buffer

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        audio_generator = audio_stream(conn)
        for iter in vad.detect_user_speech(audio_generator):
            raw_bytes = iter[0]
            is_final = iter[1]
            if raw_bytes:
                start_time = time.time()

                sf = soundfile.SoundFile(io.BytesIO(raw_bytes), channels=1, endian="LITTLE", samplerate=SAMPLING_RATE, subtype="PCM_16", format="RAW")
                audio, _ = librosa.load(sf, sr=SAMPLING_RATE)
                out.append(audio)
                out_len += len(audio)

            if (is_final or out_len >= min_sample_length) and out_len > 0:
                a = np.concatenate(out)

                # Record start time before processing

                online.insert_audio_chunk(a)

                # Record end time after processing
                

            if out_len > min_sample_length:
                o = online.process_iter()
                print('-----' * 10)
                complete_text = complete_text + o[2]
                print('PARTIAL - ' + complete_text)  # do something with current partial output
                print('-----' * 10)
                out = []
                out_len = 0
                conn.sendall(complete_text.encode('utf-8'))
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Time taken to process audio: {elapsed_time} seconds")

            if is_final:
                # Record start time before finishing processing
                start_time = time.time()

                o = online.finish()

                # Record end time after finishing processing
             

                # final_processing_pending = False
                print('-----' * 10)
                complete_text = complete_text + o[2]
                print('FINAL - ' + complete_text)  # do something with current partial output
                print('-----' * 10)
                online.init()
                out = []
                out_len = 0
                conn.sendall(complete_text.encode('utf-8'))
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Time taken to process audio: {elapsed_time} seconds")
