"""Encode a clip with several latent chunks per encoder call, the way the official tokenizer does.

``AutoencoderKLWan._encode`` calls the encoder once per latent chunk: a one-frame prime, then four pixel frames
at a time. The official stack calls that out as the reason the tokenizer under-uses the tensor cores at training
resolutions -- "each kernel launch operates on too little work to saturate the tensor cores" -- and instead
encodes a configurable number of *pixel* frames per call while carrying the causal cache across calls
(technical report Sec. 5.2.6: 68 frames at 256p, 24 at 480p, 12 at 720p; the released code's default is
``{"256": 68, "480": 24, "720": 8}``).

The result is the same latent up to floating point: the per-call caches are what make the chunking an
implementation detail, so a wider window only changes how much work each kernel launch gets. The chunk size has
to be a multiple of 4 -- the tokenizer's temporal compression -- so that the stride-2 temporal downsamples keep
their phase, which is also what the official code asserts.

    from vae_chunked import enable_chunked_encode

    enable_chunked_encode(vae, 24)          # 480p, the recipe's operating point
"""

from __future__ import annotations

import torch

# The official dict, keyed by the shorter side of the canvas. Only 480 matters for the DROID recipe.
OFFICIAL_CHUNK_FRAMES = {"256": 68, "480": 24, "720": 8}


def _encode_chunked(vae, x: torch.Tensor, chunk_frames: int, chunk_runner=None) -> torch.Tensor:
    """``AutoencoderKLWan._encode`` with a configurable encoder-call window."""
    if chunk_frames % 4:
        raise ValueError(f"chunk_frames must be a multiple of 4, got {chunk_frames}")
    _, _, num_frame, height, width = x.shape
    num_latent = (num_frame - 1) // 4 + 1

    vae.clear_cache()
    if vae.config.patch_size is not None:
        from diffusers.models.autoencoders.autoencoder_kl_wan import patchify

        x = patchify(x, patch_size=vae.config.patch_size)

    if vae.use_tiling and (width > vae.tile_sample_min_width or height > vae.tile_sample_min_height):
        return vae.tiled_encode(x)

    # Every window after the prime has to hold a whole number of 4-frame groups, or the stride-2 downsamples
    # shift phase and the result stops matching the per-chunk path. DROID's 33 frames need no padding (32 = 24
    # + 8); an arbitrary frame count does, and the extra latent frames are cropped away below.
    ragged = (num_frame - 1) % 4
    if ragged:
        x = torch.cat([x, x[:, :, -1:].expand(-1, -1, 4 - ragged, -1, -1)], dim=2)

    if chunk_runner is None:

        def run(chunk, feat_cache):
            return vae.encoder(chunk, feat_cache=feat_cache, feat_idx=[0])

    else:
        run = chunk_runner

    vae._enc_conv_idx = [0]
    out = run(x[:, :, :1, :, :], vae._enc_feat_map)
    parts = [out]
    for start in range(1, x.shape[2], chunk_frames):
        vae._enc_conv_idx = [0]
        parts.append(run(x[:, :, start : start + chunk_frames, :, :], vae._enc_feat_map))

    out = torch.cat(parts, 2) if len(parts) > 1 else parts[0]
    out = out[:, :, :num_latent]
    enc = vae.quant_conv(out)
    vae.clear_cache()
    return enc


def enable_chunked_encode(vae, chunk_frames: int, aot_dir=None, verbose: bool = False):
    """Install the wider encoder call window on ``vae``. Returns the VAE so this can wrap a constructor.

    Args:
        vae: The Wan VAE.
        chunk_frames: Pixel frames per encoder call, a multiple of 4. Zero keeps diffusers' own loop.
        aot_dir (optional): When given, compile and validate the per-chunk encoder with AOTInductor into this
            directory and dispatch to the compiled graph, falling back to eager per variant. Defaults to None,
            i.e. eager chunks.
        verbose (bool, optional): Print AOT compile/gate lines. Defaults to False.
    """
    if chunk_frames <= 0:
        return vae

    runner = None
    if aot_dir is not None:
        from vae_aot import AOTChunkRunner

        runner = AOTChunkRunner(vae.encoder, save_dir=aot_dir, verbose=verbose)

    def _encode(x):
        return _encode_chunked(vae, x, chunk_frames, runner)

    vae._encode = _encode
    vae.encode_chunk_frames = chunk_frames
    vae._aot_chunk_runner = runner
    return vae
