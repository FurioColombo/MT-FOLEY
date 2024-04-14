import argparse
import os.path
from pathlib import Path
import sys

import torch
import torchaudio as tAudio

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from eval.sample_generator import SampleGenerator

LABELS = ['DogBark', 'Footstep', 'GunShot', 'Keyboard', 'MovingMotorVehicle', 'Rain', 'Sneeze_Cough']

def list_checkpoint_paths_in_dir(dir: str or Path):
    d = os.path.abspath(dir)
    files = [os.path.abspath(os.path.join(d, f)) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) and f.split('.')[-1] == "pt"]
    return files

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Set models
    checkpoints_paths = list_checkpoint_paths_in_dir(os.path.abspath(args.checkpoints_folder_path))

    # Set sampler
    tAudio.set_audio_backend('sox_io')
    device = torch.device('cuda')
    gen = SampleGenerator(labels=LABELS, args=args, device=device)

    for path in checkpoints_paths:
        gen.make_inference(
            args=args,
            checkpoint_path=path,
        )

def validate_args(args):
    if args.gen_all_classes is False:
        assert args.class_name is not None, "if gen_all_classes is False, user should specify a class"
    else:
        assert args.class_name is None, "if all classes are being generated, no specific one should be set"

    if args.target_audio_path is None:
        print('WARNING: generating without conditioning')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints_folder_path', '-c', type=str, required=True, help="Path to the folder containing model checkpoints.")
    parser.add_argument('--param_path', '-p', type=str, default='./pretrained/params.json')
    parser.add_argument('--target_audio_path', type=str, help='Path to the target audio file.', default=None)
    parser.add_argument('--gen_all_classes', '-a', type=bool, help='Generate audio samples from all classes', default=True)
    parser.add_argument('--class_name', type=str, help='Class name to generate samples.', choices=LABELS, default=None)
    parser.add_argument('--output_dir', '-o', type=str, default="./checkpoints_results")
    parser.add_argument('--cond_scale', type=int, default=3)
    parser.add_argument('--N', type=int, default=3)
    parser.add_argument('--stereo', action='store_true',
                        help='Output stereo audio (left: target / right: generated).',
                        default=False)
    args = parser.parse_args()

    validate_args(args)
    main(args)