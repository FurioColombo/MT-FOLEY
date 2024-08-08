import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

classes = ['DogBark', 'Footstep', 'GunShot', 'Keyboard', 'MovingMotorVehicle', 'Rain', 'Sneeze_Cough']

def enum_files_in_folder():
    # path joining version for other paths
    name = 'Sneeze_Cough'
    DIR = f'./DCASE_2023_Challenge_Task_7_Dataset/dev/{name}'
    print('DIR:', DIR )
    print(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))

    DIR = f'./DCASE_2023_Challenge_Task_7_Dataset/dev/{name}_rms'
    print('DIR:', DIR )
    print(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))


def enum_txt_file_lines():
    print()
    sets = ['train', 'eval']
    for s in sets:
        file = fr"DCASE_2023_Challenge_Task_7_Dataset/{s}.txt"
        with open(file, 'r') as fp:
            x = len(fp.readlines())
            print(f'{file} - Total lines:', x)

        file = fr"DCASE_2023_Challenge_Task_7_Dataset/{s}_rms.txt"
        with open(file, 'r') as fp:
            x = len(fp.readlines())
            print(f'{file} - Total lines:', x)
        print()


def print_current_working_dir():
    # Get the current working directory
    cwd = os.getcwd()
    print(cwd)



path = os.path.abspath(print.__module__)
print(path)

'''
og_dataset_dir = os.path.abspath("./DCASE_2023_Challenge_Task_7_Dataset/eval")
out_dir = os.path.join(og_dataset_dir, 'temp_resampled_fad_background')
print('eureka')
import torch
torch.cuda.empty_cache()
#resample_wav_files_in_folder(folder_path=out_dir, target_sr=24000)
'''

print(True+1)
print(False+1)