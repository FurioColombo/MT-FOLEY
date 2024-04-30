from pathlib import Path
import argparse
import sys
import os

import numpy as np
import torchaudio
from matplotlib import pyplot as plt
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from utils.utilities import get_files_in_dir, plot_env

LABELS = ['DogBark', 'Footstep', 'GunShot', 'Keyboard', 'MovingMotorVehicle', 'Rain', 'Sneeze_Cough']
def main(args):
    if args.output_dir is not None:
        os.makedirs(os.path.abspath(args.output_dir), exist_ok=True)

    assert os.path.isdir(args.wav_path) or os.path.isfile(args.wav_path)
    if os.path.isdir(args.wav_path):
        wav_paths = get_files_in_dir(os.path.abspath(args.wav_path), extension='wav')
    elif os.path.isfile(args.wav_path):
        wav_paths = [os.path.abspath(args.wav_path), ]
    else:
        assert False, f"out_dir provided:{args.output_dir}  is invalid"

    for path in tqdm(wav_paths, desc='convert wav2img'):
        wav, sample_rate = torchaudio.load(path)
        # Convert stereo to mono if necessary
        # if len(wav.shape) > 1:
        #     wav = wav.mean(axis=1)
        wav = wav.flatten()
        # Plot the waveform
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(wav)) / sample_rate, wav)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Waveform')
        plt.grid(True)

        # Save the plot as an image file
        img_filename = os.path.basename(path).split('.')[0] + '.png'
        if args.output_dir is None:
            img_dir_name = 'wav_imgs'
            os.makedirs(os.path.join(args.wav_path, img_dir_name), exist_ok=True)
            img_filepath = os.path.join(os.path.dirname(path), img_dir_name, img_filename)
        else:
            os.makedirs(args.output_dir, exist_ok=True)
            img_filepath = args.output_dir
        plt.savefig(img_filepath)
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_path', '-in', type=str, required=True)
    parser.add_argument('--output_dir', '-o')
    args = parser.parse_args()

    main(args)