import socket
import select
import threading
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
sentence_end = 5*SAMPLING_RATE

HOST = 'localhost'
PORT = 4047

class ClientHandler:
    def __init__(self):
        self.complete_text = ''
        self.out = []
        self.out_len = 0
        self.asr = FasterWhisperASR(src_lan, model)
        self.asr.set_translate_task()
        self.tokenizer = create_tokenizer(tgt_lan)
        self.online = OnlineASRProcessor(self.asr, self.tokenizer)
        self.vad = VoiceActivityController(use_vad_result=use_vad_result)

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

def handle_client(client_handler, conn):
    print('Connected by', addr)
    audio_generator = audio_stream(conn)
    
    for iter in client_handler.vad.detect_user_speech(audio_generator):
        raw_bytes = iter[0]
        is_final = iter[1]
        
        if raw_bytes:
            sf = soundfile.SoundFile(io.BytesIO(raw_bytes), channels=1, endian="LITTLE", samplerate=SAMPLING_RATE, subtype="PCM_16", format="RAW")
            audio, _ = librosa.load(sf, sr=SAMPLING_RATE)
            client_handler.out.append(audio)
            client_handler.out_len += len(audio)

        if (is_final or client_handler.out_len >= min_sample_length) and client_handler.out_len > 0:
            a = np.concatenate(client_handler.out)
            client_handler.online.insert_audio_chunk(a)

        if client_handler.out_len > min_sample_length:
            o = client_handler.online.process_iter()
            print('-----'*10)
            client_handler.complete_text = client_handler.complete_text + o[2]
            print('PARTIAL - '+ client_handler.complete_text)  # do something with the current partial output
            print('-----'*10)     
            client_handler.out = []
            client_handler.out_len = 0
            conn.sendall(client_handler.complete_text.encode('utf-8'))

        if is_final:
            o = client_handler.online.finish()
            print('-----'*10)
            client_handler.complete_text = client_handler.complete_text + o[2]
            print('FINAL - '+ client_handler.complete_text)  # do something with the current final output
            print('-----'*10)   
            client_handler.online.init()   
            client_handler.out = []
            client_handler.out_len = 0  
            conn.sendall(client_handler.complete_text.encode('utf-8'))

    conn.close()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((HOST, PORT))
    server_socket.listen()

    print('Server listening on {}:{}'.format(HOST, PORT))

    # List to keep track of connected clients
    clients = []

    while True:
        # Use select to handle multiple clients
        readable, _, _ = select.select([server_socket] + clients, [], [])

        for s in readable:
            if s is server_socket:
                # New connection, accept it and add it to the list
                conn, addr = s.accept()
                clients.append(conn)
                # Create a new ClientHandler instance for the client
                client_handler = ClientHandler()
                # Create a new thread to handle the client
                threading.Thread(target=handle_client, args=(client_handler, conn)).start()
            else:
                # Existing client, handle it in a thread
                threading.Thread(target=handle_client, args=(client_handler, s)).start()
                # Remove the client from the list to avoid handling it again
                clients.remove(s)
