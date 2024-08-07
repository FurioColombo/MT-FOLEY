import os
import gc
import json
from glob import glob

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from torchaudio.transforms import Resample
from tqdm import tqdm

from modules.model.codec.sources.encodecWrapper import Encodec
from modules.model.codec.sources.dacWrapper import DAC
from modules.utils import notifications
from modules.utils.file_system import recursive_namespace_to_dict
from modules.utils.data_sources import dataset_from_path
from modules.model.tfmodel import UNet
from modules.model.sampler import SDESampling
from modules.model.sde import SubVpSdeCos
from modules.utils.utilities import normalize, plot_env, check_nan, check_RAM_usage
from modules.utils.audio import get_event_cond, resample_audio

LABELS = ['DogBark', 'Footstep', 'GunShot', 'Keyboard', 'MovingMotorVehicle', 'Rain', 'Sneeze_Cough']


def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: _nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)


def gc_collect():
    gc.disable()
    gc.collect()
    gc.enable()


# --- Learner ---
class Learner:
    def __init__(
        self, model_dir, model, codec:Encodec or DAC, train_set, test_set, params, device, distributed):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.device = device
        self.is_master = True
        self.distributed = distributed

        self.tensorboard_dir = os.path.join(self.model_dir, 'tensorboard') # todo: move to path manager
        self.model = model
        self.codec = codec
        self.ema_weights = [param.clone().detach()
                            for param in self.model.parameters()]
        train_params = params.training
        self.lr = train_params.lr
        self.epoch = 0
        self.step = 0
        self.valid_loss = None
        self.best_val_loss = None

        self.sde = SubVpSdeCos()
        self.ema_rate = train_params.ema_rate
        self.train_set = train_set
        self.test_set = test_set
        self.params = params
        self.use_profiler = params.profiler.use_profiler

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=train_params.scheduler_factor,
            patience=train_params.scheduler_patience_epoch * len(self.train_set) // params.logging.n_steps_to_test,
            threshold=train_params.scheduler_threshold,
        )
        self.restore_from_checkpoint(params.logging.checkpoint_id)

        self.params.total_params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.loss_fn = nn.MSELoss()
        self.v_loss = nn.MSELoss(reduction="none")
        self.summary_writer = SummaryWriter(self.tensorboard_dir, purge_step=self.step)
        self.n_bins = params.logging.n_bins
        self.num_elems_in_bins_train = torch.zeros(self.n_bins, device=device)
        self.sum_loss_in_bins_train = torch.zeros(self.n_bins, device=device)
        self.num_elems_in_bins_test = torch.zeros(self.n_bins, device=device)
        self.sum_loss_in_bins_test = torch.zeros(self.n_bins, device=device)
        self.cum_grad_norms = torch.tensor(0, device=device)

    # Train
    def train(self, profiler=None):
        device = next(self.model.parameters()).device
        while self.epoch <= self.params.training.n_epochs:
            self.train_epoch(device=device, prof=profiler)

        # notify end of training
        notification = f'TRAINING FINISHED\nepoch {self.epoch} - step {self.step}\nbest_val_loss: {self.best_val_loss}'
        notifications.notify_telegram(notification)

    def train_epoch(self, device, prof=None):
        if self.distributed: self.train_set.sampler.set_epoch(self.epoch)
        for features in (
            tqdm(self.train_set,
                 desc=f"Epoch {self.epoch}")
            if self.is_master
            else self.train_set
        ):
            if prof is not None:
                prof.step()

            self.model.train()
            features = _nested_map(
                features,
                lambda x: x.to(device) if isinstance(x, torch.Tensor) else x,
            )
            loss = self.train_step(features)
            check_nan(t=loss, error_msg=f"Detected NaN loss at step {self.step}.")

            # Logging by steps | train losses, etc
            if self.is_master:
                check_RAM_usage(self.params.max_RAM_usage, callback=lambda: self.save_to_checkpoint(filename=f'epoch-{self.epoch}'))
                if self.step % self.params.logging.n_steps_to_log == 0:
                    self._write_train_summary(self.step)

                # Validation loss
                if self.step % self.params.logging.n_steps_to_test == 0:
                   self.val_step()
            self.step += 1

        # End of epoch stuff
        if self.is_master:
            if self.epoch % self.params.logging.n_epochs_to_log == 0 and self.epoch > 0:
                # Summary writer full inference
                self._write_inference_summary(self.step, device)
                notification = f'EPOCH {self.epoch} - step {self.step}' \
                               f'\nbest_val_loss: {self.best_val_loss}'
                notifications.notify_telegram(notification)
            else:
                notifications.notify_telegram(f'Finished epoch {self.epoch}')

            # Save best model's checkpoints
            if self.epoch % self.params.logging.n_epochs_to_checkpoint == 0 or self.epoch == self.params.training.n_epochs:
                self.save_to_checkpoint(filename=f'epoch-{self.epoch}')
            self.epoch += 1
            gc_collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def val_step(self):
        self.test_set_evaluation()

        self.valid_loss = sum(self.sum_loss_in_bins_test) / sum(self.num_elems_in_bins_test)
        self._update_best_val_loss(self.valid_loss)

        self.scheduler.step(self.valid_loss)
        self.lr = self.scheduler.get_last_lr()

        self._write_test_summary(self.step)

    def train_step(self, features):
        for param in self.model.parameters():
            param.grad = None

        audio = features["audio"]
        classes = features["class"]
        events = features["event"]

        # TODO: this is temp
        # audio = self._encode_audio(audio)
        N, T = audio.shape

        t = torch.rand(N, 1, device=audio.device)
        t = (self.sde.t_max - self.sde.t_min) * t + self.sde.t_min
        noise = torch.randn_like(audio)

        noisy_audio = self.sde.perturb(audio, t, noise)
        sigma = self.sde.sigma(t)

        predicted = self.model(noisy_audio, sigma, classes, events)
        loss = self.loss_fn(noise, predicted)

        loss.backward()
        grad_norm = torch.trunc(nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)).to(torch.int64)

        self.optimizer.step()
        if self.is_master:
            self.update_ema_weights()

        t_detach = t.clone().detach()
        t_detach = torch.reshape(t_detach, (-1,))

        vectorial_loss = self.v_loss(noise, predicted).detach()
        vectorial_loss = torch.mean(vectorial_loss, 1)
        vectorial_loss = torch.reshape(vectorial_loss, (-1, ))

        self.update_conditioned_loss(vectorial_loss, t_detach, True)
        self.cum_grad_norms += grad_norm
        return loss

    @torch.no_grad()
    # Test
    def test_set_evaluation(self):
        self.model.eval()
        for features in self.test_set:
            audio = features["audio"].cuda()
            classes = features["class"].cuda()
            events = features["event"].cuda()

            N, T = audio.shape

            # TODO: this is temp
            # print('audio val_step:', type(audio), audio.device)
            # audio = self._encode_audio(audio)

            t = torch.rand(N, 1, device=audio.device)
            t = (self.sde.t_max - self.sde.t_min) * t + self.sde.t_min
            noise = torch.randn_like(audio)
            noisy_audio = self.sde.perturb(audio, t, noise)
            sigma = self.sde.sigma(t)
            predicted = self.model(noisy_audio, sigma, classes, events)

            vectorial_loss = self.v_loss(noise, predicted).detach()

            vectorial_loss = torch.mean(vectorial_loss, 1)
            vectorial_loss = torch.reshape(vectorial_loss, (-1, ))
            t = torch.reshape(t, (-1, ))
            self.update_conditioned_loss(vectorial_loss, t, False)

    # Update loss & ema weights
    def update_conditioned_loss(self, vectorial_loss, continuous_array, is_train):
        continuous_array = torch.trunc(self.n_bins * continuous_array)
        continuous_array = continuous_array.type(torch.int)
        if is_train:
            for k in range(len(continuous_array)):
                self.num_elems_in_bins_train[continuous_array[k]] += 1
                self.sum_loss_in_bins_train[continuous_array[k]] += vectorial_loss[k]
        else:
            for k in range(len(continuous_array)):
                self.num_elems_in_bins_test[continuous_array[k]] += 1
                self.sum_loss_in_bins_test[continuous_array[k]] += vectorial_loss[k]

    def update_ema_weights(self):
        for ema_param, param in zip(self.ema_weights, self.model.parameters()):
            if param.requires_grad:
                ema_param -= (1 - self.ema_rate) * (ema_param - param.detach())

    # Logging train stuff
    @torch.no_grad()
    def _write_train_summary(self, step):
        loss_in_bins_train = torch.divide(
            self.sum_loss_in_bins_train, self.num_elems_in_bins_train
        )
        dic_loss_train = {}
        for k in range(self.n_bins):
            dic_loss_train["loss_bin_" + str(k)] = loss_in_bins_train[k]

        sum_loss_n_steps = torch.sum(self.sum_loss_in_bins_train)
        mean_grad_norms = self.cum_grad_norms / self.num_elems_in_bins_train.sum() * \
                          self.params.training.batch_size

        self.summary_writer.add_scalar('train/sum_loss_on_n_steps', sum_loss_n_steps, step)
        self.summary_writer.add_scalar("train/mean_grad_norm", mean_grad_norms, step)
        self.summary_writer.add_scalars("train/conditioned_loss", dic_loss_train, step)
        self.summary_writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]['lr'], step)
        self.summary_writer.flush()
        self.num_elems_in_bins_train = torch.zeros(self.n_bins, device=self.device)
        self.sum_loss_in_bins_train = torch.zeros(self.n_bins, device=self.device)
        self.cum_grad_norms = torch.tensor(0, device=self.device)

    def _write_test_summary(self, step):
        loss_in_bins_test = torch.divide(
            self.sum_loss_in_bins_test, self.num_elems_in_bins_test
        )
        dic_loss_test = {}
        for k in range(self.n_bins):
            dic_loss_test["loss_bin_" + str(k)] = loss_in_bins_test[k]

        self.summary_writer.add_scalars("test/conditioned_loss", dic_loss_test, step)
        self.summary_writer.add_scalar("test/sum_loss_on_n_steps", torch.sum(self.sum_loss_in_bins_test), step)
        self.summary_writer.add_scalar("test/val_loss", self.valid_loss, step)
        self.summary_writer.add_scalar("test/best_val_loss", self.best_val_loss, step)
        self.summary_writer.flush()
        self.num_elems_in_bins_test = torch.zeros(self.n_bins, device=self.device)
        self.sum_loss_in_bins_test = torch.zeros(self.n_bins, device=self.device)

    def _write_inference_summary(self, step, device, cond_scale=3.):
        sde = SubVpSdeCos()
        sampler = SDESampling(self.model, sde)

        test_feature = self.test_set.dataset.get_random_sample() # todo: super slow ?
        test_event = test_feature["event"].unsqueeze(0).to(device)

        event_loss = []
        self.summary_writer.add_audio(f"test_sample/audio", test_feature["audio"], step, sample_rate=22050) #todo: is this too heavy on RAM?
        self.summary_writer.add_image(f"test_sample/envelope", plot_env(test_feature["audio"]), step, dataformats='HWC')

        for class_idx in range(len(LABELS)):
            noise = torch.randn(1, self.params.data.audio_length, device=device)
            # codec compressed audio noise
            # embedding_shape = self.codec.get_embedding_shape(batch_size=1)
            # noise = torch.randn(1, embedding_shape[-1]*embedding_shape[-2], device=device) # TODO: remove hardcoded value

            classes = torch.tensor([class_idx], device=device)

            sample = sampler.predict(noise, 100, classes, test_event, cond_scale=cond_scale)
            # sample = sample.flatten().cpu() # TODO: non latent version

            # decode audio - todo: fix this temp?
            # sample = self.codec.decode(sample, denormalize=False, unflatten=True, use_z=True)

            sample = normalize(sample)
            sample = resample_audio(sample, original_sr=24000, target_sr=22050, device=device) # todo: remove hardcoded
            sample = sample.squeeze().cpu()
            # sample = high_pass_filter(sample, sr=22050)

            event_loss.append(
                self.loss_fn(test_event.squeeze(0).cpu(), get_event_cond(sample, self.params.conditioning.event_type))
            )
            self.summary_writer.add_audio(f"{LABELS[class_idx]}/audio", sample, step, sample_rate=22050) # todo: is this too heavy on RAM?
            self.summary_writer.add_image(f"{LABELS[class_idx]}/envelope", plot_env(sample), step, dataformats='HWC')

        event_loss = sum(event_loss) / len(event_loss)
        self.summary_writer.add_scalar(f"test/event_loss", event_loss, step)
        self.summary_writer.flush()

    # Utils

    def log_params(self):
        with open(os.path.join(self.model_dir, 'params.json'), 'w') as fp:
            params_dict = recursive_namespace_to_dict(self.params)
            json.dump(params_dict, fp, indent=4)
        fp.close()

    def state_dict(self):
        if hasattr(self.model, "module") and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            "epoch": self.epoch,
            "step": self.step,
            "model": {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in model_state.items()
            },
            "optimizer": {
                k: v for k, v in self.optimizer.state_dict().items()
            },
            "scheduler": {
                k: v for k, v in self.scheduler.state_dict().items()
            },
            "ema_weights": self.ema_weights,
            "lr": self.lr,
            "best_val_loss": self.best_val_loss
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, "module") and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict["model"])
        else:
            self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.epoch = state_dict["epoch"]
        self.step = state_dict["step"] + 1
        self.ema_weights = state_dict["ema_weights"]
        self.lr = state_dict["lr"]
        self.best_val_loss = state_dict["best_val_loss"]

    def restore_from_checkpoint(self, checkpoint_id=None):
        try:
            if checkpoint_id is None:
                # find latest checkpoint_id
                list_weights = glob(f'{self.model_dir}/epoch-*.pt')
                list_ids = [int(os.path.basename(weight_path).split('-')[-1].rstrip('.pt')) for weight_path in
                            list_weights]
                checkpoint_id = list_ids.index(max(list_ids))

            checkpoint = torch.load(list_weights[checkpoint_id])
            self.load_state_dict(checkpoint)
            return True
        except (FileNotFoundError, ValueError):
            return False

    def save_to_checkpoint(self, filename="weights"):
        if self.step > 0 and self.epoch > 30:
            save_basename = f"{filename}_step-{self.step}.pt"
            save_name = f"{self.model_dir}/{save_basename}"
            print("\nsaving model to:", save_name)
            notifications.notify_telegram(f"saved model at epoch {self.epoch} - step {self.step}, with best_val_loss {self.best_val_loss}")
            torch.save(self.state_dict(), save_name)

    def _update_best_val_loss(self, val_loss):
        if self.best_val_loss is None or val_loss < self.best_val_loss:
            self.best_val_loss = self.valid_loss
            self.save_to_checkpoint(filename=f'epoch-{self.epoch}')

    def _encode_audio(self, audio): # todo: this?
        original_sr = self.params.data.sample_rate
        target_sr = self.codec.get_sample_rate()
        resampler = Resample(original_sr, target_sr, resampling_method='sinc_interp_hann').to(self.device)
        audio = resampler(audio)
        audio = audio.unsqueeze(1)
        codes, embedding =  self.codec.encode(audio, normalize_codes=True, return_embedding=True, flatten=False) # encode() requires audio with shape: (batch_size, channels, sequence_length)
        return codes


# --- Training functions ---
def _train_impl(replica_id, model, codec, train_set, test_set, params, device, distributed=False):
    torch.backends.cudnn.benchmark = True
    learner = Learner(
        params.model.model_dir, model, codec, train_set, test_set, params, device=device, distributed=distributed
    )
    learner.is_master = replica_id == 0
    learner.log_params()
    profiler_params = params.profiler
    if profiler_params.use_profiler:
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=profiler_params.wait, warmup=profiler_params.warmup,
                                             active=profiler_params.active, repeat=profiler_params.repeat),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(profiler_params.model_dir, 'tensorboard')),
            record_shapes=True,
            profile_memory=True,
            with_stack=False
        ) as prof:
            learner.train(profiler=prof)
    else:
        learner.train()


def train(params):
    model = UNet(num_classes=len(LABELS), params=params).cuda()
    codec = Encodec(sample_rate=24000, device='cuda')  # TODO: get this from ocnstructor
    # codec = DAC(sample_rate=24000, device='cuda')
    train_set = dataset_from_path(params.data.train_dirs, params, LABELS, cond_dirs=params.data.train_cond_dirs)
    test_set = dataset_from_path(params.data.test_dirs, params, LABELS, cond_dirs=params.data.test_cond_dirs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _train_impl(
        replica_id=0,
        model=model,
        codec=codec,
        train_set=train_set,
        test_set=test_set,
        params=params,
        device=device
    )

def train_distributed(replica_id, replica_count, port, params):
    print(f"Replica {replica_id} of {replica_count} started")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.distributed.init_process_group(
        "nccl", rank=replica_id, world_size=replica_count
    )
    device = torch.device("cuda", replica_id)
    torch.cuda.set_device(device)

    model = UNet(num_classes=len(LABELS), params=params).cuda()

    if params.model.codec.lower() == 'encodec':
        codec = Encodec(sample_rate=24000, device='cuda')  # TODO: get this from ocnstructor
    elif params.model.codec.lower() == 'dac':
        codec = DAC(sample_rate=24000, device='cuda')
    else:
        raise Exception(f"codec specified in params.json should either be 'dac' or 'encodec'. Found {params.model.codec}")

    train_set = dataset_from_path(params.data.train_dirs, params, LABELS, distributed=True,
                                  cond_dirs=params.data.train_cond_dirs)
    test_set = dataset_from_path(params.data.test_dirs, params, LABELS, distributed=True,
                                 cond_dirs=params.data.test_cond_dirs)
    model = DistributedDataParallel(model, device_ids=[replica_id],
                                    find_unused_parameters=True)  # todo: t-foley implementation uses find_unused_parameters=False

    _train_impl(
        replica_id=replica_id,
        model=model,
        codec=codec,
        train_set=train_set,
        test_set=test_set,
        params=params,
        device=device,
        distributed=True
    )