import os
import sys
import shutil
import soundfile as sf
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

# noinspection PyUnresolvedReferences
from frechet_audio_distance import FrechetAudioDistance
from utils.fad_utils import *

# A lightweight library for Frechet Audio Distance calculation.
# https://github.com/gudgud96/frechet-audio-distance/tree/main

# Example notebook for FAD computation
# https://github.com/gudgud96/frechet-audio-distance/blob/main/test/test_all_fad.ipynb

# EnCodec is a model trained as a neural codec, that is, it is trained to compress audio into a latent space and then reconstruct it.
# One is able to obtain high quality reconstruction from the generated embeddings.
# It encodes 1 second of audio into 75 embeddings of 128 dimensions each.

# Make folder paths
fad_path = os.path.abspath('./fad')
os.makedirs(fad_path, exist_ok=True)
encodec_path = os.path.abspath('./fad/encodec')
os.makedirs(encodec_path, exist_ok=True)

def encodec_folder_path(path: str=''):
    return os.path.join(encodec_path, path)

def fad_emb_folder_path(path: str=''):
    return os.path.join(encodec_path, 'encodings', path)

SAMPLE_RATE = 24000
LENGTH_IN_SECONDS = 1

# Specify the paths to your saved embeddings
background_embds_path = fad_path
eval_embds_path = "/path/to/saved/eval/embeddings.npy"

# create tool to measure distance
frechet = FrechetAudioDistance(
    ckpt_dir=encodec_path,
    model_name="encodec",
    # submodel_name="music_speech_audioset", # for CLAP only
    sample_rate=SAMPLE_RATE,
    # use_pca=True, # for VGGish only
    # use_activation=False, # for VGGish only
    verbose=False,
    audio_load_worker=8,
    # enable_fusion=False, # for CLAP only
)

def test_fad():
    # test FAD is working
    def gen_audio_folder_path(path: str=''):
        new_folder = encodec_folder_path('temp_test_audio')
        os.makedirs(new_folder, exist_ok=True)
        new_path = os.path.join(new_folder, path)
        return new_path

    for target, count, param in [('background', 10, None), ("test1", 5, 0.0001), ("test2", 10, 0.0000001)]:
        target = gen_audio_folder_path(target)
        os.makedirs(target, exist_ok=True)
        frequencies = np.linspace(100, 1000, count).tolist()
        for freq in frequencies:
            samples = gen_sine_wave(freq, LENGTH_IN_SECONDS, SAMPLE_RATE, noise_param=param)
            filename = os.path.join(target, "sin_%.0f.wav" % freq)
            # print("Creating: %s with %i samples." % (filename, samples.shape[0]))
            # print(os.path.abspath(filename))
            sf.write(filename, samples, SAMPLE_RATE, "PCM_24")

    fad_score = frechet.score(
        gen_audio_folder_path("background"),
        gen_audio_folder_path("test1"),
        dtype="float32"
    )
    print("FAD score test 1: %.8f" % fad_score)

    fad_score = frechet.score(
        gen_audio_folder_path("background"),
        gen_audio_folder_path("test2"),
        dtype="float32"
    )
    print("FAD score test 2: %.8f" % fad_score)

    shutil.rmtree(fad_emb_folder_path("background"))
    shutil.rmtree(fad_emb_folder_path("test1"))
    shutil.rmtree(fad_emb_folder_path("test2"))

def compute_fad(ground_truth_path, gen_audio_path, save_ground_truth_emb=False):
    fad_score = frechet.score(
        ground_truth_path,
        gen_audio_path,
        background_embds_path=background_embds_path if save_ground_truth_emb else None, # save background embeddings
        # eval_embds_path=eval_embds_path, # save audio embeddings to evaluate
        dtype="float32"
    )
    print("FAD score test 1: %.8f" % fad_score)


# ground_truth_audio_path = 'results/audio/mamba_fast_500/epoch-500_step-637037/same_class_conditioning/ground_truths/epoch-500_step-637037'
ground_truth_audio_path = 'DCASE_2023_Challenge_Task_7_Dataset/eval/DogBark'
gen_audio_path = 'results/audio/mamba_fast_500/epoch-500_step-637037/same_class_conditioning/conditioned_generation/DogBark'
compute_fad(
    ground_truth_path=ground_truth_audio_path,
    gen_audio_path=gen_audio_path,
    save_ground_truth_emb=False
)
