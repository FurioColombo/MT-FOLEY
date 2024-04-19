from pathlib import Path
import argparse
import json
import csv
import sys
import gc
import os

import torch

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from model.tfmodel import UNet
from model.sampler import SDESampling_batch, SDESampling
from model.sde import VpSdeCos
from data.dataset import from_path as dataset_from_path
from eval.checkpoint_eval import CheckpointEvaluator

LABELS = ['DogBark', 'Footstep', 'GunShot', 'Keyboard', 'MovingMotorVehicle', 'Rain', 'Sneeze_Cough']


def list_checkpoint_paths_in_dir(dir: str or Path):
    d = os.path.abspath(dir)
    files = [os.path.abspath(os.path.join(d, f)) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) and f.split('.')[-1] == "pt"]
    return files


def load_ema_weights(model, model_path):
    checkpoint = torch.load(model_path)
    dic_ema = {}
    for (key, tensor) in zip(checkpoint['model'].keys(), checkpoint['ema_weights']):
        dic_ema[key] = tensor
    model.load_state_dict(dic_ema)
    return model


def get_step_from_checkpoint_path(path: str or Path):
    return str(path).split('.')[-2].split('-')[-1]


def main(args, params):
    os.makedirs(args.output_dir, exist_ok=True)

    # get all model checkpoint paths
    if os.path.isdir(os.path.abspath(args.checkpoints_path)):
        checkpoints_paths = list_checkpoint_paths_in_dir(os.path.abspath(args.checkpoints_path))
        print('Evaluating', len(checkpoints_paths), 'checkpoints')
    elif os.path.isfile(os.path.abspath(args.checkpoints_path)):
        checkpoints_paths = [os.path.abspath(args.checkpoints_path),]
        print('Evaluating single checkpoint path:', checkpoints_paths[0])
    else:
        raise AssertionError(f'Provided checkpoints `{args.checkpoints_path}` is neither a file or directory')

    device = torch.device('cuda')

    test_set = dataset_from_path(params['test_dirs'], params, LABELS, cond_dirs=params['test_cond_dirs'])
    print('test set paths dir:', params['test_dirs'])
    checkpoint_eval = CheckpointEvaluator(
        test_set=test_set,
        labels=LABELS,
        sampler=None,
        device=device,
        audio_length=params['audio_length'],
        writer_dir=os.path.abspath(args.output_dir),
        event_type=params['event_type']
    )

    # open the file in the write mode
    csv_path = os.path.join(args.output_dir, 'checkpoints_eval.csv')
    print('csv file eval file:', csv_path)

    with open(csv_path, 'w') as f:
        # create the csv writer
        csv_writer = csv.writer(f, dialect='excel')

        # define model and sampling
        model = UNet(len(LABELS), params).to(device)
        sde = VpSdeCos()

        for path in checkpoints_paths:
            step = get_step_from_checkpoint_path(path)
            if int(step) >= args.start_step:
                print('evaluating: ', path)
                # prepare model and sampling
                model = load_ema_weights(model, os.path.abspath(path))
                sampler = SDESampling(model, sde)
                checkpoint_eval.set_sampler(sampler)

                # test checkpoints
                event_loss = checkpoint_eval.test_checkpoint_inference(
                    step=step,
                    cond_scale=args.cond_scale,
                    sampler=sampler
                )

                # save results
                # write a row to the csv file
                row = [step, event_loss]
                csv_writer.writerow(row)
                gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints_path', '-c', type=str, required=True)
    parser.add_argument('--param_path', '-p', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    parser.add_argument('--cond_scale', type=int, default=3)
    parser.add_argument('--N', type=int, default=3)
    parser.add_argument('--start_step', '-s', default=0, type=int)
    args = parser.parse_args()

    with open(args.param_path) as f:
        params = json.load(f)

    main(args, params)