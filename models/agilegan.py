import torch
from torch import nn
from models.encoders import vae_encoder
from models.stylegan2.model import Generator


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class EncoderModel(nn.Module):

    def __init__(self, opts):
        super(EncoderModel, self).__init__()
        self.set_opts(opts)
        # Define architecture
        self.encoder = vae_encoder.VAEStyleEncoder(50, self.opts)
        self.decoder = Generator(1024, 512, 8)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights()

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading model from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
            self.__load_latent_avg(ckpt)
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
        else:
            # print('Loading encoders weights from irse50!')
            # encoder_ckpt = torch.load(model_paths['ir_se50'])
            # # if input to encoder is not an RGB image, do not load the input layer weights
            # if self.opts.label_nc != 0:
            #     encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
            # self.encoder.load_state_dict(encoder_ckpt, strict=False)
            # print('Loading decoder weights from pretrained!')
            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
            if self.opts.learn_in_w:
                self.__load_latent_avg(ckpt, repeat=1)
            else:
                self.__load_latent_avg(ckpt, repeat=1)  # self.opts.n_styles

    def forward(self, x, resize=True, randomize_noise=True, return_latents=False, alpha=None):

        codes, logvar, mu = self.encoder(x)
        latent = [self.decoder.style(s) for s in codes]
        latent = [torch.stack(latent, dim=0)]

        input_is_latent = True
        images, result_latent = self.decoder(latent,
                                             input_is_latent=input_is_latent,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents)

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent, mu, logvar
        else:
            return images

    def set_opts(self, opts):
        self.opts = opts

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None
