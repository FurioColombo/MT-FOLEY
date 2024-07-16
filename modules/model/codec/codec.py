from abc import ABC, abstractmethod
import torch

from modules.utils.utilities import map_to_range


class Codec(ABC):
    def __init__(self, model, sample_rate, device, codebook_min, codebook_max, n_codebooks, codebook_t_steps, emb_dim):
        self.model = model
        self.sample_rate = sample_rate
        self.device = device
        self.codebook_min = codebook_min
        self.codebook_max = codebook_max
        self.n_codebooks = n_codebooks
        self.codebook_t_steps = codebook_t_steps # todo: compute it from frame per sec
        self.emb_dim = emb_dim

    @abstractmethod
    def encode(self, audio_data):
        pass

    @abstractmethod
    def decode(self, encoded_data):
        pass

    def to(self, device):
        self.model.to(device)

    def normalize_codebook(self, z, z_min=0, z_max=1):
        return map_to_range(
            z,
            min_val=self.codebook_min,
            max_val=self.codebook_max,
            new_min=z_min,
            new_max=z_max
        )

    def denormalize_notebook(self, z, z_min=0, z_max=1):
        return map_to_range(
            z,
            min_val=z_min,
            max_val=z_max,
            new_min=self.codebook_min,
            new_max=self.codebook_max
        )

    def apply_notebook_bounds(self, codebooks):
        # Set values less than 0 to 0
        codebooks = torch.where(codebooks < self.codebook_min, torch.tensor(self.codebook_min), codebooks) # todo: suspicious
        # Set values greater than 1023 to 1023
        codebooks = torch.where(codebooks > self.codebook_max, torch.tensor(self.codebook_max), codebooks) # todo: suspicious
        return codebooks

    def get_codes_shape(self, batch_size=1):
        return batch_size, self.n_codebooks, self.codebook_t_steps

    def get_embedding_shape(self, batch_size=1):
        return batch_size, self.emb_dim, self.codebook_t_steps