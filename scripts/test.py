import os

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