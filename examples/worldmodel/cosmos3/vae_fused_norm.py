"""A cheaper RMSNorm for the frozen Wan VAE: same normalisation, three fewer passes over the activation.

``WanRMS_norm.forward`` in diffusers 0.40 evaluates, for a half-precision input::

    normalized = F.normalize(x.float(), dim=1).to(x.dtype)      # up-cast, divide, down-cast
    return normalized * self.scale * self.gamma + self.bias     # two multiplies and an add

``self.bias`` is the Python float 0.0 in this checkpoint (the module is built with ``bias=False``), so the add is a
full-size kernel that adds nothing; and the two multiplies can be folded into the division by pre-dividing the
*scalar* instead of the tensor.

The replacement keeps the reduction in fp32 -- that is the part that has to stay accurate -- but computes
``scale / norm`` on the small ``[B, 1, T, H, W]`` tensor, casts that once, and leaves one broadcast multiply per
affine term. The per-channel ``gamma`` cannot be folded into the following convolution: every one of these norms
feeds a SiLU first (``WanResidualBlock.forward``), which is not homogeneous. So the pass count per norm is:

    original   up-cast, divide, down-cast, *scale, *gamma, +0.0     (6 full passes)
    fused      *scale/norm, *gamma                                  (2 full passes)

Numerically the fused form rounds the shared ``scale / norm`` factor to the input dtype before multiplying, where
the original rounds the quotient per element. That adds at most one rounding of the input dtype per element; the
gate is the same as everywhere else in this work -- stay under the bf16 cast that ``packing.py`` applies anyway.
"""

from __future__ import annotations

import types

import torch


def _fused_forward(self, x: torch.Tensor) -> torch.Tensor:
    dim = 1 if self.channel_first else -1
    dtype = x.dtype
    # `dtype=` casts inside the reduction instead of materialising an fp32 copy of the activation.
    norm = torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=True, dtype=torch.float32).clamp_min(1e-12)
    # The clamp keeps a degenerate all-zero window from turning `scale / norm` into inf before it meets the zero:
    # diffusers divides the tensor first (0 / 1e-12 = 0), so the fused order has to saturate instead of overflow.
    scale = (self.scale / norm).clamp_max(torch.finfo(dtype).max).to(dtype)  # [B, 1, T, H, W]: small
    out = x * scale
    out.mul_(self.gamma)  # in place: `out` is already a fresh tensor, and this saves a full write
    if isinstance(self.bias, torch.Tensor):
        out = out + self.bias
    return out


def enable_fused_rmsnorm(vae, verbose: bool = False):
    """Patch every ``WanRMS_norm`` inside ``vae`` with the fused forward. Returns the VAE."""
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm

    patched = 0
    for module in vae.modules():
        if isinstance(module, WanRMS_norm):
            module.forward = types.MethodType(_fused_forward, module)
            patched += 1
    vae.fused_rmsnorm_modules = patched
    if verbose:
        print(f"[vae] fused RMSNorm in {patched} module(s)", flush=True)
    return vae
