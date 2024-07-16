import dac
import torch
from audiotools import AudioSignal

# Download a model
model_path = dac.utils.download(model_type="24khz")
DAC = dac.DAC.load(model_path)
DAC.to('cuda')

# Load audio signal file
from encodec.utils import convert_audio

signal = AudioSignal("./DCASE_2023_Challenge_Task_7_Dataset/eval/DogBark/043.wav")

signal = signal.resample(24000)
# Encode audio signal as one long file
# (may run out of GPU memory on long files)
signal.to(DAC.device)


x = DAC.preprocess(signal.audio_data, signal.sample_rate)
z, codes, latents, _, _ = DAC.encode(x)
z_info = (z.shape, z.dtype)
codes_info = (codes.shape, codes.dtype)
latents_info = (latents.shape, latents.dtype)
wav = DAC.decode(z)
wav_info = (wav.shape, wav.dtype)


print('\nz1:')
for j, i in enumerate(z1):
    print(f' -{j}: {type(i), i.shape}')

# Decode audio signal
y = model.decode(z)
print('y', type(y), y.shape)

y1 = model.decode(z1[0])
print('y1', type(y1), y1.shape)

print('\n',y==y1)
print(y)
print(y1)

exit()
# Alternatively, use the `compress` and `decompress` functions
# to compress long files.

signal = signal.cpu()
x = model.compress(signal)

# Save and load to and from disk
# x.save("compressed.dac")
x = dac.DACFile.load("compressed.dac")

# Decompress it back to an AudioSignal
y = model.decompress(x)

# Write to file
#y y.write('output.wav')