from pathlib import Path
from tqdm import tqdm
import logging

import argparse
import os.path
import sys

import torch
import torchaudio as tAudio

sys.path.append(str(Path(__file__).parent.parent.parent.parent.absolute()))
from modules.eval.sample_generator import SampleGenerator
from modules.utils.utilities import check_RAM_usage, load_json_config
from modules.utils.file_system import ProjectPaths
from modules.utils.notifications import notify_telegram

'''
Usage example:
 
python ./scripts/inference/inference_multi_checkpoints.py \
    -m logs/mamba_fast/epoch-500_step-637037.pt \
    -p logs/mamba_fast/params.json \
    -a True \
    -o results/audio/mamba_fast_500/epoch-500_step-637037/conditioned_100/ \
    -s True \
    -n 10 
'''

LABELS = ['DogBark', 'Footstep', 'GunShot', 'Keyboard', 'MovingMotorVehicle', 'Rain', 'Sneeze_Cough']


def list_checkpoint_paths_in_dir(dir: str or Path):
    d = os.path.abspath(dir)
    files = [os.path.abspath(os.path.join(d, f)) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) and f.split('.')[-1] == "pt"]
    return files

def prepare_machine():
    # Set model and sampler
    tAudio.set_audio_backend('sox_io')

def main(args):
    config = load_json_config(ProjectPaths().config_file)

    prepare_machine()
    os.makedirs(args.output_dir, exist_ok=True)

    # Set models
    if os.path.isfile(args.checkpoints_path):
        checkpoints_paths = [os.path.abspath(args.checkpoints_path)]
    elif os.path.isdir(args.checkpoints_path):
        checkpoints_paths = list_checkpoint_paths_in_dir(os.path.abspath(args.checkpoints_path))
    else:
        raise IsADirectoryError(f'checkpoint path {args.checkpoints_path} is invalid')

    # Set sampler
    tAudio.set_audio_backend('sox_io')
    device = torch.device('cuda')
    gen = SampleGenerator(
        labels=LABELS,
        args=args,
        device=device,
    )

    for path in tqdm(checkpoints_paths, desc=f"checkpoints inference"):
        check_RAM_usage(config.max_RAM_usage)
        gen.make_inference(
            args=args,
            checkpoint_path=path,
            same_class_conditioning=args.same_class_conditioning
        )

    notify_telegram(f'Finished script execution: {__file__}')

def validate_args(args):
    if args.gen_all_classes is False:
        assert args.class_names is not None, "if gen_all_classes is False, user should specify a class"
    else:
        assert args.class_names is None, "if all classes are being generated, no specific one should be set"

    if args.target_audio_path is None:
       # todo: implement this
       logging.warning('WARNING: generating without conditioning')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints_path', '-m', type=str, required=True, help="Path to the folder containing model checkpoints.")
    parser.add_argument('--param_path', '-p', type=str, default='./pretrained/params.json')
    parser.add_argument('--target_audio_path', '-t', type=str, help='Path to the target audio file.', default=None)
    parser.add_argument('--gen_all_classes', '-a', type=bool, help='Generate audio samples from all classes', default=False)
    parser.add_argument('--class_names', '-c', nargs='+', type=str, required=False, help='Class name to generate samples.',
                        choices=LABELS, default=None)
    parser.add_argument('--output_dir', '-o', type=str, default="./checkpoints_results")
    parser.add_argument('--same_class_conditioning', '-s', type=bool, default=False)
    parser.add_argument('--cond_scale', type=int, default=3)
    parser.add_argument('--N', '-n', type=int, default=3)
    parser.add_argument('--stereo', action='store_true',
                        help='Output stereo audio (left: target / right: generated).',
                        default=False)
    args = parser.parse_args()

    validate_args(args)
    main(args)
