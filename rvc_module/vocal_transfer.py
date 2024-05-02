import torch
from rvc_module import RVC
import load_audio

def process_audio(audio_path):
    rvc_webui = RVC()

    model_path = './models/zeroshot.pth'
    model = torch.load(model_path)
    rvc_webui.set_model(model)
    
    audio = load_audio(audio_path)

    output = rvc_webui.run_model(audio)


    return output

# audio_path = 'temp_hanzala.wav'
# output = process_audio(audio_path)
# print(output)