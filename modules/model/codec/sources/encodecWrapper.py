import os
import torch
from einops import rearrange, unpack
from encodec import EncodecModel

from modules.model.codec.codec import Codec
from modules.utils.utilities import normalize, map_to_range, normalize_to_range

# Took some inspiration from here: https://github.com/lucidrains/audiolm-pytorch/blob/main/audiolm_pytorch/encodec.py

class Encodec(Codec):
    '''
    TODO:
    - see if we need to keep the scaled version and somehow persist the scale factors for when we need to decode? Right
        now I'm just setting self.model.normalize = False to sidestep all of that
    '''

    def __init__(self, sample_rate, device, band_width=6.0):

        # Additional initialization if needed for MyCodec
        encodec_weights_path = os.path.abspath('./pretrained/encodec')
        torch.hub.set_dir(encodec_weights_path)

        # Encodec model setup
        if sample_rate == 24000:
            model = EncodecModel.encodec_model_24khz() # Instantiate a pretrained EnCodec model
        elif sample_rate == 48000:
            model = EncodecModel.encodec_model_48khz() # TODO: bandwidth in 48KHz case
        else:
            raise Exception(f'Trying to set sr to: {sample_rate}.\n.Encodec codec only supports 24kHz and 44kHz.')
        # todo: rmv hardcoding
        super().__init__(model, sample_rate, device,
                         codebook_min=0, codebook_max=1024, n_codebooks=8, codebook_t_steps=300, emb_dim=128)

        # The number of codebooks used will be determined by the bandwidth selected.
        # E.g. for a bandwidth of 6kbps, `n_q = 8` codebooks are used.
        # Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
        # For the 48 kHz model, only 3, 6, 12, and 24 kbps are supported. The number
        # of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.
        self.model.set_target_bandwidth(band_width)
        self.model.normalize = False
        self.model.to(self.device)
        self.num_quantizers = self.get_num_quantizers()
        self.emb_frames_per_second = 75

    @torch.no_grad()
    def encode(self, audio_tensor, normalize_codes=True, return_embedding=True, flatten=False):
        codes =  self.model.encode(audio_tensor)
        # Comment from audiolm - but perfect for our case
        # encoded_frames is a list of (frame, scale) tuples. Scale is a scalar but we don't use it. Frame is a tensor
        # of shape [batch, num_quantizers, num_samples_per_frame].
        # NOT YET - We want to concatenate the frames to get all the timesteps concatenated.
        codes = torch.cat([encoded[0] for encoded in codes], dim=-1)  # [B, n_q, T]
        batch_size = codes.shape[0]

        # compute embedding z
        emb = None
        if return_embedding:
            emb = self.model.quantizer.decode(rearrange(codes, 'b q t -> q b t'))

        codes = codes.float().to(torch.float32)  # todo: suspicious

        if normalize:
            codes = self.normalize_codebook(codes)

        if flatten:
            codes = rearrange(codes, 'b q t -> b (q t)')
            emb = rearrange(emb, 'b q t -> b (q t)')

        return codes, emb

    @torch.no_grad()
    def decode(self, audio_encoding_tensor, denormalize=True, unflatten=True, use_codes:bool=False, use_z:bool=False): # todo:improve architecture here
        assert use_z is not use_codes, 'codec encode method should either receive latent codes `z` or codes `codes`'

        if use_codes:
            return self._decode_codes(audio_encoding_tensor, denormalize, unflatten)
        elif use_z:
            return self._decode_z(audio_encoding_tensor, unflatten)

    def _decode_codes(self, audio_encoding_tensor, denormalize=True, unflatten=True):
        if denormalize:
            audio_encoding_tensor = self.denormalize_notebook(audio_encoding_tensor)

        # Set values less than 0 to 1024
        self.apply_notebook_bounds(audio_encoding_tensor)

        discrete_codes = audio_encoding_tensor.int().to(torch.int64)

        # format variable to the shape needed for encoding
        if unflatten:
            discrete_codes = discrete_codes.view(discrete_codes.size(0), self.n_codebooks,
                                                 self.codebook_t_steps)  # todo: remove hardcoding

        # format for model function | list(tuple(tensor, None))
        formatted_discrete_codes = [(discrete_codes, None), ]

        return self.model.decode(formatted_discrete_codes)

    def _decode_z(self, z, unflatten:bool = False):
        # format variable to the shape needed for encoding
        if unflatten:
            z = z.view(z.size(0), self.emb_dim, self.codebook_t_steps)  # todo: remove hardcoding
        return self.model.decoder(z)

    def get_channels(self):
        return self.model.channels

    def get_sample_rate(self):
        return self.model.sample_rate

    # hacky way to get num quantizers (from audiolm)
    def get_num_quantizers(self, audio_length=512):
        assert self.model is not None, "Before calling this function you need to initialize self.model"
        out = self.model.encode(torch.randn(1, 1, audio_length, device=self.device))
        return out[0][0].shape[1]

    # taken from audiolm
    def get_emb_from_indices(self, indices):
        # todo: debug this
        codes = rearrange(indices, 'b t q -> q b t')
        emb = self.model.quantizer.decode(codes)
        return rearrange(emb, 'b c n -> b n c')
