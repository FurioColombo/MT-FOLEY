import os
import torch
import dac

from modules.model.codec.codec import Codec
from modules.utils.utilities import map_to_range

class DAC(Codec):
    # B: Batch size
    # N:
    # T:
    def __init__(self, sample_rate, device, band_width=6.0):
        if sample_rate == 16000:
            model_code = '16khz'
        elif sample_rate == 24000:
            model_code = '24khz'
        elif sample_rate == 44000:
            model_code = '44khz'
        else:
            raise Exception(f'Trying to set sr to: {sample_rate}.\n.dac codec only supports 16kHz, 24kHz and 44kHz.')
        # Additional initialization if needed for MyCodec
        dac_weights_path = os.path.abspath('./pretrained/dac')
        torch.hub.set_dir(dac_weights_path)

        # Download a model
        model_path = dac.utils.download(model_type=model_code)
        model = dac.DAC.load(model_path)
        # todo: rmv hardcoding
        super().__init__(model, sample_rate, device,
                         codebook_min=0, codebook_max=1023, n_codebooks=32, codebook_t_steps=300, emb_dim=128)

        self.model.to(self.device)

    @torch.no_grad()
    def _encode(self, audio, normalize_codes=True, return_embeddings=True, flatten=False):
        # DAC encode doc:  Returns
        #             "codes" : Tensor[B x N x T]
        #                 Codebook indices for each codebook
        #                 (quantized discrete representation of input)

        z, codes, latent, _, _ = self.model.encode(audio)
        batch_size = codes.shape[0]

        if flatten:
            codes = codes.view(batch_size, -1)
            z = z.view(batch_size, -1)

        if normalize_codes:
            codes = map_to_range(codes, min_val=self.codebook_min, max_val=self.codebook_max, new_min=0, new_max=1)

        if return_embeddings is False:
            z = None

        return codes, z

    def _encode_lantent(self, audio, normalize=True):
        # DAC encode doc:  Returns
        #             "latents" : Tensor[B x N*D x T]
        #                 Projected latents (continuous representation of input before quantization)

        _, _, latents, _, _ = self.model.encode(audio)
        if normalize:
            latents = map_to_range(latents, min_val=self.codebook_min, max_val=self.codebook_max, new_min=0, new_max=1)
        return latents

    @torch.no_grad()
    def decode(self, audio_encoding_tensor, denormalize=True):
        # [0, 1] range to 0, 1024
        if denormalize:
            audio_encoding_tensor = self.denormalize_notebook(audio_encoding_tensor)

        # Set values less than 0 to 1024
        audio_encoding_tensor = self.apply_notebook_bounds(audio_encoding_tensor)

        discrete_codes = audio_encoding_tensor.int().to(torch.int64)

        discrete_codes = self.apply_notebook_bounds(discrete_codes)

        # Compute quantized continuous representation of input
        discrete_codes = discrete_codes.view(discrete_codes.size(0), self.n_codebooks, self.codebook_t_steps) # todo: remove hardcoding
        z, latent, codes = self.model.quantizer.from_codes(discrete_codes)

        # Decode audio signal
        return self.model.decode(z)

    def get_channels(self):
        return self.model.channels

    def get_sample_rate(self):
        return self.model.sample_rate

