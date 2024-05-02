import socket
import sounddevice as sd
import numpy as np
import select

def record_audio(duration=1):
    sample_rate = 16000
    channels = 1
    dtype = np.int16

    # Record audio
    audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=channels, dtype=dtype)
    sd.wait()
    return audio_data.tobytes()

def send_audio_stream(server_address=('2.tcp.ngrok.io', 15754)):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        try:
            # Connect to the server
            client_socket.connect(server_address)
            print('connected')
            
            # Set the socket to non-blocking mode
            client_socket.setblocking(0)

            # Initialize transcript
            transcript = ""
            while True:
                # Record audio chunk
                raw_audio_data = record_audio()  
                
                # Send the raw audio data
                client_socket.sendall(raw_audio_data)

                # Check for available data to receive
                ready_to_read, _, _ = select.select([client_socket], [], [], 0.1)

                if ready_to_read:
                    # Receive and print both raw response and processed transcript
                    response = client_socket.recv(2048)
                    new_words = response.decode('utf-8').replace('\n', '').replace('\r', '')
                    print(new_words)               
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Send audio stream to the NGROK server
    send_audio_stream()
