import csv
import os

import numpy as np
import torch
import torchaudio
from torch.utils.data.distributed import DistributedSampler

from utils import get_event_cond


def parse_filelist(filelist_path):
    # if filelist_path is txt file
    if filelist_path.endswith('.txt'):
        with open(filelist_path, 'r') as f:
            filelist = [line.strip() for line in f.readlines()]
        return filelist

    # if filelist_path is csv file
    if filelist_path.endswith('.csv'):
        with open(filelist_path, 'r') as f:
            reader = csv.reader(f)
            filelist = [row[0] for row in reader]
            f.close()
        return filelist


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, paths, params, labels):
        super().__init__()
        self.filenames = []
        self.audio_length = params['audio_length']
        self.labels = labels
        self.event_type = params['event_type']
        for path in paths:
            self.filenames += parse_filelist(path)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        signal, _ = torchaudio.load(audio_filename)
        signal = signal[0, :self.audio_length]
            
        # extract class cond
        cls_name = os.path.dirname(audio_filename).split('/')[-1]
        cls = torch.tensor(self.labels.index(cls_name))
        
        # extract event cond
        event = signal.clone().detach()
        event = get_event_cond(event, self.event_type)
        
        return {
            'audio': signal,
            'class': cls,
            'event': event
        }
        
    def moving_avg(self, input, window_size):
        if type(input) != list: input = list(input)
        result = []
        for i in range(1, window_size+1):
            result.append(sum(input[:i])/i)
        
        moving_sum = sum(input[:window_size])
        result.append(moving_sum/window_size)
        for i in range(len(input) - window_size):
            moving_sum += (input[i+window_size] - input[i])
            result.append(moving_sum/window_size)
        return np.array(result)
    


class CondAudioDataset(torch.utils.data.Dataset):
    def __init__(self, audio_paths, params, labels, cond_paths=None):
        super().__init__()
        self.audio_filenames = []
        self.cond_filenames = []
        self.audio_length = params['audio_length']
        self.labels = labels
        self.event_type = params['event_type']
        self.load_conditioning = audio_paths is not None

        if self.load_conditioning:
            for audio_path, cond_path in zip(audio_paths, cond_paths):
                self.audio_filenames += parse_filelist(audio_path)
                self.cond_filenames += parse_filelist(cond_path)

    def __len__(self):
        return len(self.audio_filenames)

    def __getitem__(self, idx):
        audio_filename = self.audio_filenames[idx]
        signal, _ = torchaudio.load(audio_filename)
        signal = signal[0, :self.audio_length]

        # extract class cond
        cls_name = os.path.dirname(audio_filename).split('/')[-1]
        cls = torch.tensor(self.labels.index(cls_name))

        cond_filename = self.cond_filenames[idx]
        event = torch.load(cond_filename)

        return {
            'audio': signal,
            'class': cls,
            'event': event
        }


def from_path(data_dirs, params, labels, distributed=False, cond_dirs=None):
    if cond_dirs is None:
        dataset = AudioDataset(data_dirs, params, labels)
    else:
        print('Conditioning is being loaded from file!')
        dataset = CondAudioDataset(
            audio_paths=data_dirs,
            cond_paths=cond_dirs,
            params=params,
            labels=labels
        )

    if distributed:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=params['batch_size'],
            collate_fn=None,
            shuffle=False,
            num_workers=params['num_workers'],
            pin_memory=True,
            drop_last=True,
            sampler=DistributedSampler(dataset)
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params['batch_size'],
        collate_fn=None,
        shuffle=True,
        num_workers=os.cpu_count()//4,
        pin_memory=True,
        drop_last=True
    )
