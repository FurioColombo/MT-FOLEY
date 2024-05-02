import sys
import json
import argparse
from builtins import int
from pathlib import Path

import torch
import torchaudio as T
from torchsummary import summary

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from model.tfmodel import UNet
LABELS = ['DogBark', 'Footstep', 'GunShot', 'Keyboard', 'MovingMotorVehicle', 'Rain', 'Sneeze_Cough']


def load_ema_weights(model, model_path):
    checkpoint = torch.load(model_path)
    dic_ema = {}
    for (key, tensor) in zip(checkpoint['model'].keys(), checkpoint['ema_weights']):
        dic_ema[key] = tensor
    model.load_state_dict(dic_ema)
    return model

def main(args):
    # Set model and sampler
    T.set_audio_backend('sox_io')
    device = torch.device('cuda')

    with open(args.param_path) as f:
        params = json.load(f)
    model = UNet(len(LABELS), params).to(device)

    summary(model, depth=args.depth)
    print('========================= END =========================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m', type=str, default='./pretrained/block-49_epoch-500.pt')
    parser.add_argument('--param_path', '-p', type=str, default='./pretrained/params.json')
    parser.add_argument('--depth', '-d', type=int, default=1)
    args = parser.parse_args()

    main(args)