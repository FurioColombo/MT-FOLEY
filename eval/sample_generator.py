import os
import json
from pathlib import Path
import random

import torch
import numpy as np
import torchaudio as tAudio

import pydub
import soundfile as sf
from scipy.io.wavfile import write

from model.tfmodel import UNet
from model.sampler import SDESampling_batch
from model.sde import VpSdeCos
from data.dataset import from_path as dataset_from_path
from utils.utilities import adjust_audio_length, get_event_cond, high_pass_filter, normalize, resample_audio

def load_ema_weights(model, model_path):
    checkpoint = torch.load(model_path)
    dic_ema = {}
    for (key, tensor) in zip(checkpoint['model'].keys(), checkpoint['ema_weights']):
        dic_ema[key] = tensor
    model.load_state_dict(dic_ema)
    return model


def save_samples(samples, output_dir, sr, class_name, stereo=False, target_audio=None):
    for j in range(samples.shape[0]):
        sample = samples[j].cpu()
        sample = high_pass_filter(sample)
        write(f"{output_dir}/{class_name}_{str(j + 1).zfill(3)}.wav", sr, sample)

        if stereo:
            assert target_audio is not None, "Target audio is required for stereo output."
            left_audio = target_audio.cpu().numpy()
            right_audio = sample.copy()
            assert len(left_audio) == len(right_audio), "Length of target and generated audio must be the same."

            sf.write('temp_left.wav', left_audio, 22050, 'PCM_24')
            sf.write('temp_right.wav', right_audio, 22050, 'PCM_24')

            left_audio = pydub.AudioSegment.from_wav('temp_left.wav')
            right_audio = pydub.AudioSegment.from_wav('temp_right.wav')

            if left_audio.sample_width > 4:
                left_audio = left_audio.set_sample_width(4)
            if right_audio.sample_width > 4:
                right_audio = right_audio.set_sample_width(4)

            # pan the sound
            left_audio_panned = left_audio.pan(-1.)
            right_audio_panned = right_audio.pan(+1.)

            mixed = left_audio_panned.overlay(right_audio_panned)
            mixed.export(f"{output_dir}/{class_name}_{str(j + 1).zfill(3)}_stereo.wav", format='wav')

            # remove temp files
            os.remove('temp_left.wav')
            os.remove('temp_right.wav')


def measure_el1_distance(sample, target, event_type):
    sample = normalize(sample).cpu()
    target = normalize(target).cpu()

    sample_event = get_event_cond(sample, event_type)
    target_event = get_event_cond(target, event_type)

    # sample_event = pooling(sample_event, block_num=49)
    # target_event = pooling(target_event, block_num=49)

    loss_fn = torch.nn.L1Loss()
    loss = loss_fn(sample_event, target_event)
    return loss.cpu().item()


class SampleGenerator:
    def __init__(self, labels:list, args, device):
        self.labels = labels
        self.device = device
        self.n_gen_samples_per_class = args.N
        self.stereo = args.stereo
        self.results_dir = args.output_dir

        # load params
        with open(args.param_path) as f:
            self.params = json.load(f)
        self.sample_rate = self.params['sample_rate']
        self.audio_length = self.sample_rate * 4

        self.target_audio = None
        self.target_event = None
        self.update_conditioning(args.target_audio_path)

        self.test_set = None

    def make_inference(self, args, checkpoint_path: str or Path, same_class_conditioning=False):
        self.update_conditioning(args.target_audio_path)

        # load model
        model = UNet(len(self.labels), self.params).to(self.device)
        model = load_ema_weights(model, os.path.abspath(checkpoint_path))
        sde = VpSdeCos()
        sampler = SDESampling_batch(model, sde, batch_size=self.n_gen_samples_per_class , device=self.device)

        # Generate N samples
        if args.gen_all_classes:
            class_indices = range(len(self.labels))
        else:
            class_indices = [self.labels.index(args.class_name)]

        if same_class_conditioning:
            self.test_set = dataset_from_path(self.params['test_dirs'], self.params, self.labels, cond_dirs=self.params['test_cond_dirs'])

        generated = self._generate_samples(
            class_indices = class_indices,
            sampler = sampler,
            cond_scale = args.cond_scale,
            checkpoint_path = checkpoint_path,
            same_class_conditioning=same_class_conditioning
        )
        print('Done!')

        # Measure E-L1 distance if target audio is given
        if args.target_audio_path is not None:
            dists = []
            for sample in generated:
                dist = measure_el1_distance(sample, self.target_audio, self.params['event_type'])
                dists.append(dist)
            print(f"E-L1 distance: {np.mean(dists)}")

    def _generate_samples(self, class_indices:list, sampler, cond_scale, checkpoint_path, same_class_conditioning: False):
        def _compute_conditioning():
            if same_class_conditioning:
                target_audio = self.get_random_test_sample(class_index=class_idx)
                target_event = target_audio["event"].unsqueeze(0).to(self.device)
            else:
                target_event = self.target_event
                target_audio = self.target_audio

            return target_audio, target_event

        def _generate_conditioned_samples():
            # print(f"Generate {self.n_gen_samples_per_class} samples of class \'{self.labels[class_idx]}\'...")
            noise = torch.randn(self.n_gen_samples_per_class, self.audio_length, device=self.device)
            classes = torch.tensor([class_idx] * self.n_gen_samples_per_class, device=self.device)
            sampler.batch_size = self.n_gen_samples_per_class
            samples = sampler.predict(noise, 100, classes, target_event, cond_scale=cond_scale)

            out_dir = self.compute_out_dir(self.results_dir, checkpoint_path=checkpoint_path)
            save_samples(
                samples,
                out_dir,
                self.sample_rate,
                self.labels[class_idx],
                self.stereo,
                target_audio
            )
            return samples

        generated_samples = []
        for class_idx in class_indices:
            target_audio, target_event = _compute_conditioning()
            class_samples = _generate_conditioned_samples()
            generated_samples.append(class_samples)
        return generated_samples

    def update_conditioning(self, cond_audio_path):
        # Prepare target audio for conditioning (if exist)
        if cond_audio_path is not None:
            target_audio, sr = tAudio.load(cond_audio_path)
            if sr != self.sample_rate:
                target_audio = resample_audio(target_audio, sr, self.sample_rate)
            self.target_audio = adjust_audio_length(target_audio, self.audio_length)
            self.target_event = get_event_cond(target_audio, self.params['event_type'])
            self.target_event = self.target_event.repeat(self.n_gen_samples_per_class, 1).to(self.device)

    def remove_conditioning(self):
        self.target_audio = None
        self.target_event = None

    def compute_out_dir(self, results_dir, checkpoint_path, category: str=None):
        # results directory
        results_dir = os.path.abspath(results_dir)
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)

        # epoch inference directory
        epoch_results_dir_name = checkpoint_path.split('.')[0].split('/')[-1]
        directory = os.path.join(results_dir, epoch_results_dir_name)
        if not os.path.isdir(directory):
            os.mkdir(directory)

        # category inference directory
        if category is not None:
            directory = os.path.join(directory, category)
            if not os.path.isdir(directory):
                os.mkdir(directory)

        return  directory


    def get_random_test_sample(self, class_index: int=None): # TODO: make this a dataset class
        if class_index is None:
            return self.test_set.dataset[random.choice(range(len(self.test_set.dataset)))]
        else:
            test_set_category =  self.test_set.labels[class_index]
            return test_set_category.dataset[random.choice(range(len(test_set_category.dataset)))]
