import argparse
import json
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from model import UNet
from sampler import SDESampling, SDESampling_batch
from sde import SubVpSdeCos, VpSdeCos
from utils import plot_env, normalize, high_pass_filter, get_event_cond

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

def test_checkpoint_inference(sampler, step, device, cond_scale=3.):
    test_feature = self.get_random_test_feature()
    test_event = test_feature["event"].unsqueeze(0).to(device)

    event_loss = []
    writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
    writer.add_audio(f"test_sample/audio", test_feature["audio"], step, sample_rate=22050)
    writer.add_image(f"test_sample/envelope", plot_env(test_feature["audio"]), step, dataformats='HWC')

    for class_idx in range(len(LABELS)):
        noise = torch.randn(1, self.params['audio_length'], device=device)
        classes = torch.tensor([class_idx], device=device)

        sample = sampler.predict(noise, 100, classes, test_event, cond_scale=cond_scale)
        sample = sample.flatten().cpu()

        sample = normalize(sample)
        sample = high_pass_filter(sample, sr=22050)

        event_loss.append(self.loss_fn(test_event.squeeze(0).cpu(), get_event_cond(sample, self.params['event_type'])))
        writer.add_audio(f"{LABELS[class_idx]}/audio", sample, step, sample_rate=22050)
        writer.add_image(f"{LABELS[class_idx]}/envelope", plot_env(sample), step, dataformats='HWC')

    event_loss = sum(event_loss) / len(event_loss)
    writer.add_scalar(f"test/event_loss", event_loss, step)
    writer.flush()

    def test_checkpoint(sampler, test_dataset):



def main(args):
    # get all model checkpoint paths
    checkpoints_paths = list_checkpoint_paths_in_dir(os.path.abspath(args.checkpoints_folder_path))

    device = torch.device('cuda')
    with open(args.param_path) as f:
        params = json.load(f)

    for path in checkpoints_paths:
        # load model
        model = UNet(len(LABELS), params).to(device)
        model = load_ema_weights(model, os.path.abspath(path))
        sde = VpSdeCos()
        sampler = SDESampling_batch(model, sde, batch_size=args.N, device=device)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./pretrained/block-49_epoch-500.pt')
    parser.add_argument('--param_path', type=str, default='./pretrained/params.json')
    parser.add_argument('--target_audio_path', type=str, help='Path to the target audio file.', default=None)
    parser.add_argument('--class_name', type=str, required=True, help='Class name to generate samples.',
                        choices=LABELS)
    parser.add_argument('--output_dir', type=str, default="./results")
    parser.add_argument('--cond_scale', type=int, default=3)
    parser.add_argument('--N', type=int, default=3)
    parser.add_argument('--stereo', action='store_true',
                        help='Output stereo audio (left: target / right: generated).',
                        default=False)
    args = parser.parse_args()

    main(args)