"""Two exact changes to the Wan encoder's convolution path: fewer copies, same arithmetic.

1. **H/W padding goes to the kernel.** ``WanCausalConv3d.forward`` calls ``F.pad(x, (1, 1, 1, 1, 2, 0))`` before every
   3x3x3 convolution -- a full-size copy that on Ascend shows up as ``aclnnConstantPadNd`` (227 launches, 5.1% of the
   bf16 profile). Only the temporal pad is asymmetric (two frames in front, none behind: that is what makes the
   convolution causal); the H/W pad is symmetric, which is exactly what ``F.conv3d``'s own ``padding`` argument
   means. In steady state the cache supplies the two frames in front, so ``pad_t`` is zero and nothing is copied at
   all; only the prime call of an encode still pads T.

2. **The residual-down-block clone is dead.** ``WanResidualDownBlock.forward`` starts with ``x_copy = x.clone()``
   so that ``avg_shortcut`` can read the input after the residual stack has rebound ``x``. Nothing writes into the
   input in place, so the shortcut can be computed first and the clone dropped: on the widest block that is a
   128 MB copy per call, four blocks per encoder call.

Both are arithmetically exact -- the tests assert bit equality against diffusers' own forward -- so they carry no
accuracy cost at all, unlike the fused RMSNorm which rounds ``scale / norm`` once before multiplying.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F


def _fast_conv_forward(self, x, cache_x=None):
    pad_t = self._padding[4]
    if cache_x is not None and pad_t > 0:
        cache_x = cache_x.to(x.device)
        x = torch.cat([cache_x, x], dim=2)
        pad_t -= cache_x.shape[2]
    if pad_t > 0:
        # Causal: frames in front, none behind. H/W are the convolution's own job.
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
    return F.conv3d(
        x,
        self.weight,
        self.bias,
        self.stride,
        (0, self._padding[2], self._padding[0]),
        self.dilation,
        self.groups,
    )


def _fast_down_forward(self, x, feat_cache=None, feat_idx=[0]):
    shortcut = self.avg_shortcut(x)
    for resnet in self.resnets:
        x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
    if self.downsampler is not None:
        x = self.downsampler(x, feat_cache=feat_cache, feat_idx=feat_idx)
    return x + shortcut


def enable_fast_conv(vae, verbose: bool = False):
    """Give every 3x3x3 convolution its H/W padding instead of materialising it."""
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

    patched = 0
    for module in vae.modules():
        if isinstance(module, WanCausalConv3d):
            module.forward = types.MethodType(_fast_conv_forward, module)
            patched += 1
    vae.fast_conv_modules = patched
    if verbose:
        print(f"[vae] conv H/W padding handed to the kernel in {patched} module(s)", flush=True)
    return vae


def enable_fast_downblock(vae, verbose: bool = False):
    """Drop the clone in every residual down block."""
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanResidualDownBlock

    patched = 0
    for module in vae.modules():
        if isinstance(module, WanResidualDownBlock):
            module.forward = types.MethodType(_fast_down_forward, module)
            patched += 1
    vae.fast_downblock_modules = patched
    if verbose:
        print(f"[vae] residual-down-block clone dropped in {patched} block(s)", flush=True)
    return vae


def enable_fast_encoder(vae, verbose: bool = False):
    enable_fast_conv(vae, verbose)
    enable_fast_downblock(vae, verbose)
    return vae


def disable_fast_encoder(vae):
    """Restore diffusers' own forwards."""
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d, WanResidualDownBlock

    for module in vae.modules():
        if isinstance(module, (WanCausalConv3d, WanResidualDownBlock)) and "forward" in module.__dict__:
            del module.__dict__["forward"]
    return vae
