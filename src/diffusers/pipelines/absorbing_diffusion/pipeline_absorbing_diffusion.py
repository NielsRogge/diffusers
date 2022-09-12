import warnings
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.distributions as dists

from torchvision import utils

from ...models import Transformer, VQModel
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput


def latent_ids_to_onehot(latent_ids, latent_shape, codebook_size):
    min_encoding_indices = latent_ids.view(-1).unsqueeze(1)
    encodings = torch.zeros(min_encoding_indices.shape[0], codebook_size).to(latent_ids.device)
    encodings.scatter_(1, min_encoding_indices, 1)
    one_hot = encodings.view(latent_ids.shape[0], latent_shape[1], latent_shape[2], codebook_size)
    return one_hot.reshape(one_hot.shape[0], -1, codebook_size)


class AbsorbingDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for unconditional image generation using Absorbing Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`VQModel`]):
            Vector-Quantized Variational Auto-Encoder (VQ-VAE) Model to encode and decode images to and from discrete
            latent representations.
        transformer ([`Transformer`]):
            BERT-like Transformer encoder.
        mask_id (`int`, *optional*, defaults to 1024):
            The id of the mask token in the vocabulary.
    """

    def __init__(
        self,
        vae: VQModel,
        transformer: Transformer,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            transformer=transformer,
        )
        # TODO make these attributes configurable?
        self.mask_id = 1024
        self.latent_shape = [1, 16, 16]
        self.codebook_size = self.transformer.vocab_size
        self.emb_dim = self.vae.latent_channels

    @torch.no_grad()
    def embed(self, z):
        z_flattened = z.view(-1, self.codebook_size)  # B*H*W, codebook_size
        embedded = (
            torch.matmul(z_flattened, self.vae.quantize.embedding.weight)
            .view(z.size(0), self.latent_shape[1], self.latent_shape[2], self.emb_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return embedded

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
        num_inference_steps: int = 256,
        temp=1.0,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                Number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `nd.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """

        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            # Set device as before (to be removed in 0.3.0)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        x_t = torch.ones((batch_size, np.prod(self.latent_shape)), device=self.device).long() * self.mask_id
        unmasked = torch.zeros_like(x_t, device=self.device).bool()

        sample_steps = list(range(1, num_inference_steps + 1))

        for t in reversed(sample_steps):
            print("Time step:", t)
            t = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # where to unmask
            changes = torch.rand(x_t.shape, generator=generator, device=self.device) < 1 / t.float().unsqueeze(-1)
            # don't unmask somewhere already unmasked
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
            # update mask with changes
            unmasked = torch.bitwise_or(unmasked, changes)

            print("Shape of Transformer input:", x_t.shape)
            print("First values:", x_t[0, :3])

            x_0_logits = self.transformer(x_t, t=t)

            print("Shape of x_0_logits:", x_0_logits.shape)
            print("First values of logits:", x_0_logits[0, :3, :3])

            # scale by temperature
            x_0_logits = x_0_logits / temp
            x_0_dist = dists.Categorical(logits=x_0_logits)
            x_0_hat = x_0_dist.sample().long()
            x_t[changes] = x_0_hat[changes]

        # decode the image latents with the VAE decoder
        latents = x_t
        latents_one_hot = latent_ids_to_onehot(latents, self.latent_shape, self.codebook_size)
        q = self.embed(latents_one_hot)
        images = self.vae.decoder(q.float())

        # if output_type == "pil":
        #     images = self.numpy_to_pil(images)
        utils.save_image(images, "results.png", nrow=batch_size)

        if not return_dict:
            return (images,)

        return ImagePipelineOutput(images=images)
