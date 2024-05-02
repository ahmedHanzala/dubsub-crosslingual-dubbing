import argparse
from seq2seq_translation.urdu_translate_60_bleu import translate
from pydub import AudioSegment
from whisper.transcribe_whisper import transcribe
from tts_module.urdu_finetuned.main import generate_audio
from rvc_module.vocal_transfer import process_audio
from wav2lip.inference import sync_lips_with_audio


def extract_audio(video_path):
    video = AudioSegment.from_file(video_path)
    audio = video.audio
    return audio


def chunk_audio(audio, timestamps):
    chunks = []
    for start, end, transcription in timestamps:
        chunk = audio[start:end]
        chunk_transcription = transcription
        chunks.append({'audio': chunk, 'transcription': chunk_transcription})
    return chunks

def merge_audio(processed_audio_chunks,video_length,audio_from_vid):
    merged_audio = processed_audio_chunks[0]['audio']
    for i in range(1, len(processed_audio_chunks)):
        merged_audio += processed_audio_chunks[i]['audio']

        # Pick remaining audio from source audio
        if len(merged_audio) < video_length:
            remaining_audio = audio_from_vid[len(merged_audio):]
            merged_audio += remaining_audio
    return merged_audio


def main():
    parser = argparse.ArgumentParser(description='Process video path, CUDA device number, and video length.')

    parser.add_argument('video_path', type=str, help='path to the video file')
    parser.add_argument('cuda_device', type=int, help='CUDA device number')
    parser.add_argument('video_length', type=int, help='length of the video in milliseconds')

    args = parser.parse_args()

    video_path = args.video_path
    cuda_device = args.cuda_device
    video_length = args.video_length

    audio_from_vid = extract_audio(video_path)
    source_transcription_timestamps = transcribe(audio_from_vid, timestamps = True)

    ## chunk audio wrt to timestammps do not send all audio over to TTS module
    audio_chunks = chunk_audio(audio=audio_from_vid,source_transcription_timestamps = source_transcription_timestamps)

    processed_audio_chunks = []

    for object in audio_chunks:
        translated_text, timestamps = translate(object['transcription'],True)
        translated_audio  = generate_audio(translated_text)
        ## apply vocal transfer
        vocal_transfer_output = process_audio(translated_audio)

        item = {
            "audio": vocal_transfer_output,
            "timestamps": timestamps
        }
        processed_audio_chunks.append(item) 

    ## merge audio chunks
    final_audio = merge_audio(processed_audio_chunks,video_length,audio_from_vid)
    #print(len(final_audio) == len(audio_from_vid))
    final_vid = sync_lips_with_audio(final_audio,video_path)
    final_vid.export("final_video.mp4",format="mp4")

    
    ## video and final audio --> wav2lip --> output

        # Save merged audio to file
        # with open('hanzala.wav', 'wb') as f:
        #     f.write(merged_audio.export(format='wav'))
        # with open('hanzala.wav', ' w') as f:
        #     f.write(vocal_transfer_output)
                




if __name__ == '__main__':
    main()