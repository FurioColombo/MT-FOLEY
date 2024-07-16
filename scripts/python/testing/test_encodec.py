import os
import sys
from pathlib import Path

import torch
import torchaudio

from encodec.utils import convert_audio

sys.path.append(str(Path(__file__).parent.parent.parent.parent.absolute()))
from modules.model.codec.sources.encodecWrapper import Encodec
from modules.utils.audio import high_pass_filter, get_event_cond


encodec_weights_path = os.path.abspath('./pretrained/encodec')
torch.hub.set_dir(encodec_weights_path)
print(torch.hub.get_dir())

'''
# Instantiate a pretrained EnCodec model
model = EncodecModel.encodec_model_24khz()
# The number of codebooks used will be determined by the bandwidth selected.
# E.g. for a bandwidth of 6kbps, `n_q = 8` codebooks are used.
# Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
# For the 48 kHz model, only 3, 6, 12, and 24 kbps are supported. The number
# of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.
model.set_target_bandwidth(6.0)

# Load and pre-process the audio waveform
wav, sr = torchaudio.load("./DCASE_2023_Challenge_Task_7_Dataset/eval/DogBark/042.wav")
wav = convert_audio(wav, sr, model.sample_rate, model.channels)
wav = wav.unsqueeze(0)

# Extract discrete codes from EnCodec
with torch.no_grad():
    encoded_frames = model.encode(wav)
    print('encoded_frames:', type(encoded_frames), 'len', len(encoded_frames))
    for i, a in enumerate(encoded_frames):
        print(f'    - elem {i}', type(a), len(a))
        for j , b in enumerate(a):
            print(f'        - elem {j}', type(b))
            if hasattr(b, 'shape'):
                print('         shape:', b.shape)
    # print(np.array(encoded_frames).shape)

codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]


print(codes)
print(codes.shape)
'''


# ================================================================================
'''
model = Encodec(
    sample_rate=24000,
    device='cpu'
)
model.to('cuda')
print('script directory: ', os.path.abspath(os.path.curdir))
print('===================================================================================================', end='\n\n')
# Load and pre-process the audio waveform
wav, sr = torchaudio.load(os.path.abspath("./DCASE_2023_Challenge_Task_7_Dataset/eval/DogBark/043.wav"))
wav_og = wav
wav = convert_audio(wav, sr, model.get_sample_rate(), model.get_channels())
wav = wav.unsqueeze(0)
wav = wav.to('cuda')
encoded_frames = model.encode(wav, normalize=True)

noise = [torch.rand_like(encoded_frames) / i for i in range(1, 1201, 150)]
# noisy_embeddings = [encoded_frames + n for n in noise]
a = encoded_frames[:, :2100]
noisy_embeddings = [torch.cat((encoded_frames[:, :2100], n[:, 2100:]), -1) for n in noise]

print('diff')
for n in noisy_embeddings:
    print(torch.mean(torch.abs(n - encoded_frames)))

reconstructed_noisy_wavs = [model.decode(j, denormalize=True) for j in noisy_embeddings]
reconstructed_wav = model.decode(encoded_frames, denormalize=True)

print('MAE')
print('clean: ', torch.mean(torch.abs(wav - reconstructed_wav)))
print('noisy:')
for rec in reconstructed_noisy_wavs:
    print(torch.mean(torch.abs(rec - wav)))
print()
reconstructed_noisy_wavs = [noisy[0] for noisy in reconstructed_noisy_wavs]
reconstructed_noisy_wavs = [normalize(noisy) for noisy in reconstructed_noisy_wavs]
reconstructed_wav = normalize(reconstructed_wav[0])
print('MSE')
print('clean: ', torch.mean(torch.square(wav - reconstructed_wav)))
print('noisy:')
for idx, rec in enumerate(reconstructed_noisy_wavs):
    print(torch.mean(torch.square(rec - wav)))
    torchaudio.save(f'./results/codec/test_encodec_audio_noisy_{idx}.wav', rec.to('cpu'), sample_rate=24000)
print()
# reconstructed_wav = high_pass_filter(reconstructed_wav[0], sr=24000)

# save audios
torchaudio.save('./results/codec/test_encodec_gen_audio_clean.wav', reconstructed_wav.to('cpu'), sample_rate=24000)
torchaudio.save('./results/codec/test_encodec_original_audio.wav', wav[0].to('cpu'), sample_rate=24000)

'''

# ==================================== DEBUG FOR SHAPES ====================================
def encodec_embeddings_shapes():
    from encodec import EncodecModel
    wav, sr = torchaudio.load(os.path.abspath("./DCASE_2023_Challenge_Task_7_Dataset/eval/DogBark/043.wav"))
    model = EncodecModel.encodec_model_24khz().to('cuda')
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0)
    wav = wav.to('cuda')

    from encodec import EncodecModel
    model = EncodecModel.encodec_model_24khz().to('cuda')
    # Load and pre-process the audio waveform
    wav_shape = wav.shape
    encoded_frames  = model.encode(wav)
    info0 = (type(encoded_frames), len(encoded_frames))
    info1 = (type(encoded_frames[0]), len(encoded_frames[0]))
    info2 = (encoded_frames[0][0].shape, encoded_frames[0][0].dtype)

    rec_wav = model.decode(encoded_frames)
    info_wav = (rec_wav.shape, rec_wav.dtype)

    print()

encodec_embeddings_shapes()
