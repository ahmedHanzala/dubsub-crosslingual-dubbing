import time
import socket
from microphone_stream import MicrophoneStream
import threading
import os
import argparse

SAMPLING_RATE = 16000

def receive_data(client_socket):
    while True:
        try:
            data = client_socket.recv(1024).decode()
            # end = time.time()
            # print('end time: ', end)

            if os.name == 'nt':
                _ = os.system('cls')
            else:
                _ = os.system('clear')
            print(data, end='', flush=True)
            # elapsed = end - start
            # print('start time: ', start)
            # print(f"Time taken to process audio: {elapsed} seconds")
        except socket.error as e:
            print(f"Error receiving data: {e}")
            break

def main():
    parser = argparse.ArgumentParser(description="Microphone Streaming Client")
    parser.add_argument("--ip", type=str, help="Server IP address", required=True)
    parser.add_argument("--port", type=int, help="Server port", required=True)
    args = parser.parse_args()

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((args.ip, args.port))
    microphone_stream = MicrophoneStream(sample_rate=SAMPLING_RATE)

    # Create a thread for receiving data
    receive_thread = threading.Thread(target=receive_data, args=(client_socket,), daemon=True)
    receive_thread.start()

    try:
        for raw_bytes in microphone_stream:
            client_socket.sendall(raw_bytes)
            # global start
            # start = time.time()

    except KeyboardInterrupt:
        pass
    finally:
        client_socket.close()
        microphone_stream.close()

if __name__ == "__main__":
    main()