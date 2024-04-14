import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import os

from data.dataset import parse_filelist
from params.params import params
from utils.utils import get_event_cond


def create_cond_files_from_txt(paths_files):
    # for each file containing paths:
    for paths_file in paths_files:
        print('paths_file', paths_file)
        audio_paths = parse_filelist(paths_file)

        # open/create txt file for storing paths
        txt_file_path, _ = os.path.splitext(os.path.normpath(paths_file))
        cond_txt_file_path = txt_file_path + f"_{params['event_type']}" + ".txt"
        if os.path.isfile(cond_txt_file_path):
            os.remove(cond_txt_file_path)
            print('Deleted file at', cond_txt_file_path)

        print(f'Creating conditioning files from {paths_file} into dev folder') # todo: logs
        with open(cond_txt_file_path, 'w') as f:
            print(f'Create new conditioning paths file .txt at {cond_txt_file_path}')

            for audio_path in tqdm(audio_paths):
                audio_path = Path(audio_path)
                signal, _ = torchaudio.load(audio_path)
                signal = signal[0, :params['audio_length']]

                # extract event cond
                cond = get_event_cond(signal, params['event_type'])

                # check for cond files folder existence
                cond_folder_path = Path(str(Path(audio_path).parent) + f"_{params['event_type']}")
                if not os.path.isdir(cond_folder_path):
                    os.mkdir(cond_folder_path)
                    print('Created', cond_folder_path, 'directory')

                # save conditioning file .pt
                cond_file_name = audio_path.stem + '.pt'
                cond_path = os.path.join(cond_folder_path, cond_file_name)
                torch.save(cond, cond_path)
                f.write(f'{cond_path}\n')

txt_files = params['train_dirs']
create_cond_files_from_txt(txt_files)

txt_files = params['test_dirs']
create_cond_files_from_txt(txt_files)