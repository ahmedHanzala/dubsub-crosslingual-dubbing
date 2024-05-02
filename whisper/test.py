from pydub import AudioSegment
import numpy as np

def raw_to_mp3(raw_file, mp3_file, sample_width=2, channels=1, frame_rate=16000):
    # Read raw audio data
    with open(raw_file, 'rb') as f:
        raw_data = f.read()

    # Convert raw data to numpy array
    audio_array = np.frombuffer(raw_data, dtype=np.int16)

    # Reshape audio array to the correct shape
    audio_array = audio_array.reshape((-1, channels))

    # Create an AudioSegment from the numpy array
    audio_segment = AudioSegment(
        audio_array.tobytes(),
        frame_rate=frame_rate,
        sample_width=sample_width,
        channels=channels
    )

    # Export the AudioSegment to an MP3 file
    audio_segment.export(mp3_file, format="mp3")

if __name__ == "__main__":
    raw_file_path = "recorded_audio.raw"
    mp3_file_path = "recorded_audio.mp3"

    raw_to_mp3(raw_file_path, mp3_file_path)
    print(f"Conversion completed. MP3 file saved at {mp3_file_path}")
