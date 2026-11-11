# The network architecture in this file is adapted from video-diffusion-pytorch
# by Phil Wang:
# https://github.com/lucidrains/video-diffusion-pytorch
# The original code is distributed under the following license.
#
# MIT License
#
# Copyright (c) 2022 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
from functools import partial

import torch
from torch import einsum, nn
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
from einops_exts import rearrange_many
from rotary_embedding_torch import RotaryEmbedding

from utilities.data_consistency.mri import (
    H_operator,
    PatchConverter,
    SOFT_SENSE_GAMMA_S,
)
from utilities.data_consistency.patches import TEMPORAL_BOUNDARY_MODES
from utilities.reproducibility import make_torch_generator, randn_like, seed_everything


def exists(value):
    return value is not None


def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# relative positional bias

class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads = 8,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype = torch.long, device = device)
        k_pos = torch.arange(n, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.scale * self.gamma

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding = (0, 1, 1))
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)

class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, _, _, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h = self.heads)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b = b)

# attention along space and time

class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        rotary_emb = None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

    def forward(
        self,
        x,
        pos_bias = None,
        focus_present_mask = None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim = -1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h = self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device = device, dtype = torch.bool)
            attend_self_mask = torch.eye(n, device = device, dtype = torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)

# model

class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        attn_heads = 8,
        attn_dim_head = 32,
        init_dim = None,
        init_kernel_size = 7,
        use_sparse_linear_attn = True,
    ):
        super().__init__()
        self.channels = channels

        # temporal attention and its relative positional encoding

        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        temporal_attn = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(dim, heads = attn_heads, dim_head = attn_dim_head, rotary_emb = rotary_emb))

        self.time_rel_pos_bias = RelativePositionBias(heads = attn_heads, max_distance = 32) # realistically will not be able to generate that many frames of video... yet

        # initial conv

        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size, init_kernel_size), padding = (0, init_padding, init_padding))

        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        block_klass = ResnetBlock
        time_conditioned_block = partial(block_klass, time_emb_dim = time_dim)

        # modules for all layers

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                time_conditioned_block(dim_in, dim_out),
                time_conditioned_block(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(dim_out, heads = attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = time_conditioned_block(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(mid_dim, heads = attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))

        self.mid_block2 = time_conditioned_block(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                time_conditioned_block(dim_out * 2, dim_in),
                time_conditioned_block(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(dim_in, heads = attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

    def forward(self, x, time, focus_present_mask=None):

        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device = x.device)

        x = self.init_conv(x)

        x = self.init_temporal_attn(x, pos_bias = time_rel_pos_bias)

        r = x.clone()

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
        x = self.mid_block2(x, t)

        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
            x = upsample(x)

        x = torch.cat((x, r), dim = 1)
        return self.final_conv(x)

def make_beta_schedule(schedule, n_timestep, **kwargs):
    """Create the diffusion beta schedule used by the reconstruction checkpoint."""
    if schedule != "linear":
        raise ValueError("CineDiff reconstruction supports only the linear beta schedule")
    linear_start = kwargs.get("beta_start", 0.0001)
    linear_end = kwargs.get("beta_end", 0.02)
    return torch.linspace(
        linear_start,
        linear_end,
        n_timestep,
        dtype=torch.float64,
    )


class GaussianDiffusion(nn.Module):
    """Diffusion state and the single CineDiff reconstruction sampler."""

    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_frames,
        channels=3,
        timesteps=1000,
        beta_scheduler="linear",
        **kwargs,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn
        betas = make_beta_schedule(beta_scheduler, timesteps, **kwargs)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.num_timesteps = int(betas.shape[0])

        def register_float_buffer(name, value):
            self.register_buffer(name, value.to(torch.float32))

        # Keep every historical schedule buffer because they are part of the
        # strict checkpoint state dict, even when the sampler reads only a subset.
        register_float_buffer("betas", betas)
        register_float_buffer("alphas_cumprod", alphas_cumprod)
        register_float_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        register_float_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_float_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_float_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        register_float_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        register_float_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod - 1),
        )

        posterior_variance = (
            betas
            * (1.0 - alphas_cumprod_prev)
            / (1.0 - alphas_cumprod)
        )
        register_float_buffer("posterior_variance", posterior_variance)
        register_float_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_float_buffer(
            "posterior_mean_coef1",
            betas
            * torch.sqrt(alphas_cumprod_prev)
            / (1.0 - alphas_cumprod),
        )
        register_float_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - alphas_cumprod),
        )

        self.sigma_data = 0.07674478632312676

    def EDM_denoiser(self, x, sigma):
        """Apply the EDM preconditioning used by the trained denoiser."""
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1, 1)

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = (
            sigma
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2).sqrt()
        )
        c_in = 1.0 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        prediction = self.denoise_fn(
            c_in * x,
            c_noise.flatten(),
        ).to(torch.float32)
        return c_skip * x + c_out * prediction

    def ddim_sample(
        self,
        shape,
        *,
        rand_seed=None,
        eta=None,
        gamma_w=None,
        gamma_s=SOFT_SENSE_GAMMA_S,
        num_sense_maps=None,
        NFE_tgt=None,
        input_data_folder=None,
        strides=None,
        temporal_boundary_mode="balanced_full",
    ):
        """Run the seeded Euler/DDIM transition with DPS data consistency."""
        self.last_soft_sense_components = None
        if rand_seed is None:
            raise ValueError("rand_seed is required")
        rand_seed = seed_everything(rand_seed)
        if temporal_boundary_mode not in TEMPORAL_BOUNDARY_MODES:
            raise ValueError(
                "temporal_boundary_mode must be 'legacy_pad' or 'balanced_full', "
                f"got {temporal_boundary_mode!r}"
            )

        sampling_timesteps = NFE_tgt
        if not isinstance(eta, (int, float)) or isinstance(eta, bool) or not 0 <= eta <= 1:
            raise ValueError("eta must be a number between 0 and 1")
        if (
            not isinstance(gamma_w, (int, float))
            or isinstance(gamma_w, bool)
            or not math.isfinite(gamma_w)
            or gamma_w < 0
        ):
            raise ValueError("gamma_w must be a finite, non-negative number")
        if (
            not isinstance(gamma_s, (int, float))
            or isinstance(gamma_s, bool)
            or not math.isfinite(gamma_s)
            or gamma_s < 0
        ):
            raise ValueError("gamma_s must be a finite, non-negative number")
        if not isinstance(sampling_timesteps, int) or isinstance(sampling_timesteps, bool):
            raise ValueError("NFE_tgt must be an integer")
        if not 2 <= sampling_timesteps <= self.num_timesteps:
            raise ValueError(
                f"NFE_tgt must be between 2 and {self.num_timesteps}"
            )
        total_timesteps = self.num_timesteps
        device = self.betas.device
        noise_generator = make_torch_generator(rand_seed, device)

        if sampling_timesteps == total_timesteps:
            times = torch.linspace(
                -1,
                total_timesteps - 1,
                steps=sampling_timesteps + 1,
            )
        else:
            times = torch.linspace(
                -1,
                total_timesteps - 1,
                steps=total_timesteps + 1,
            )[:sampling_timesteps]

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        # Preserve the established RNG stream: shortened schedules consume this
        # initial draw before replacing it with the noisy CS initialization.
        z_t_previous = torch.randn(
            shape,
            device=device,
            generator=noise_generator,
        )

        measurement_operator = H_operator(
            device=device,
            input_data_folder=input_data_folder,
            num_sense_maps=num_sense_maps,
        )
        if strides is None:
            raise ValueError("strides is required")
        patch_converter = PatchConverter(
            data_shape=measurement_operator.image_shape,
            strides=strides,
            temporal_boundary_mode=temporal_boundary_mode,
        )

        coherent_noise_generator = None

        def patch_noise_like(reference):
            nonlocal coherent_noise_generator
            if temporal_boundary_mode == "legacy_pad":
                return randn_like(reference, generator=noise_generator)
            if coherent_noise_generator is None:
                coherent_noise_generator = make_torch_generator(rand_seed, device)
            height, width, frames = measurement_operator.image_shape
            full_noise = torch.randn(
                (2, height, width * frames),
                dtype=reference.dtype,
                device=reference.device,
                generator=coherent_noise_generator,
            )
            patch_noise = patch_converter.i2p(full_noise)
            if tuple(patch_noise.shape) != tuple(reference.shape):
                raise RuntimeError(
                    "coherent temporal noise does not match the sampler shape: "
                    f"expected {tuple(reference.shape)}, got {tuple(patch_noise.shape)}"
                )
            return patch_noise.to(reference.dtype)

        for index, (time, time_next) in enumerate(
            tqdm(
                time_pairs,
                desc="sampling loop time step",
                ncols=80,
            )
        ):
            alpha = self.alphas_cumprod[time].to(torch.float64)
            alpha_next = self.alphas_cumprod[time_next].to(torch.float64)

            if index == 0 and sampling_timesteps != total_timesteps:
                initial_noise_std = torch.sqrt((1.0 - alpha) / alpha)
                cs_image = measurement_operator.cs_init.squeeze(0).to(device)
                cs_patches = patch_converter.i2p(
                    cs_image,
                    offset=[0, 0, 0],
                )
                z_t_previous = (
                    cs_patches
                    + patch_noise_like(cs_patches) * initial_noise_std
                )
            elif index == 0:
                # Full EDM start: p_T = sigma_T * n. Sigma is the standard
                # deviation, so independent n gives covariance sigma_T**2 * I.
                # balanced_full retains shared noise in overlapping patches.
                initial_noise_std = torch.sqrt((1.0 - alpha) / alpha)
                initial_noise = z_t_previous
                if temporal_boundary_mode == "balanced_full":
                    initial_noise = patch_noise_like(z_t_previous)
                z_t_previous = initial_noise * initial_noise_std

            if index == 0 and temporal_boundary_mode == "balanced_full":
                with torch.no_grad():
                    z_t_previous = patch_converter.project_consensus(z_t_previous)

            if time_next < 0:
                with torch.no_grad():
                    sigma_t = torch.sqrt((1.0 - alpha) / alpha)
                    z_start = self.EDM_denoiser(
                        z_t_previous.detach(),
                        sigma_t,
                    ).clamp(-1.0, 1.0)
                    if temporal_boundary_mode == "balanced_full":
                        z_start = patch_converter.project_consensus(z_start)
                    if measurement_operator.map_set_count > 1:
                        final_primary = patch_converter.p2i(z_start).unsqueeze(0)
                        measurement_operator.update_auxiliary_components(
                            final_primary,
                            data_weight=gamma_w,
                            l1_weight=gamma_s,
                        )
                        self.last_soft_sense_components = (
                            measurement_operator.components_from_primary(final_primary)
                            .detach()
                            .to("cpu")
                        )
                return unnormalize_img(z_start)

            with torch.enable_grad():
                z_t = z_t_previous.detach().requires_grad_(True)
                sigma_t = torch.sqrt((1.0 - alpha) / alpha)
                z_start = self.EDM_denoiser(z_t, sigma_t).clamp(-1.0, 1.0)

                # The DDIM transition does not participate in the DPS gradient.
                # Avoid retaining its unused autograd graph.
                noise = patch_noise_like(z_t)
                with torch.no_grad():
                    # Additive-noise DDIM: stochastic_sigma = varsigma_k,
                    # transition_scale = h_k, denoised_scale = g_k = 1 - h_k.
                    sigma_sq = (1.0 - alpha) / alpha
                    sigma_next_sq = (1.0 - alpha_next) / alpha_next
                    stochastic_sigma = eta * torch.sqrt(
                        torch.clamp(
                            sigma_next_sq * (sigma_sq - sigma_next_sq) / sigma_sq,
                            min=0.0,
                        )
                    )
                    transition_scale = torch.sqrt(
                        torch.clamp(
                            (sigma_next_sq - stochastic_sigma**2) / sigma_sq,
                            min=0.0,
                        )
                    )
                    denoised_scale = 1.0 - transition_scale
                    z_prev = transition_scale * z_t + denoised_scale * z_start
                    if time != 0:
                        z_prev = z_prev + stochastic_sigma * noise

                image_pred = patch_converter.p2i(z_start).unsqueeze(0)
                measurement_pred = measurement_operator.H(image_pred)
                measurement_error = torch.linalg.norm(
                    (measurement_pred - measurement_operator.y).squeeze(),
                    ord=2,
                )
                error = measurement_error * gamma_w
                gradients = torch.autograd.grad(error, inputs=z_t)[0]
                if temporal_boundary_mode == "balanced_full":
                    gradients = patch_converter.pullback_temporal_gradient(gradients)
                z_prev = z_prev - gradients
                if temporal_boundary_mode == "balanced_full":
                    with torch.no_grad():
                        z_prev = patch_converter.project_consensus(z_prev)
                if measurement_operator.map_set_count > 1:
                    measurement_operator.update_auxiliary_components(
                        image_pred.detach(),
                        data_weight=gamma_w,
                        l1_weight=gamma_s,
                    )

            z_t_previous = z_prev.detach()

        raise RuntimeError("sampling schedule ended without a terminal denoising step")

    def sample(self, batch_size=16, **kwargs):
        """Reconstruct a batch with the sole supported sampler."""
        shape = (
            batch_size,
            self.channels,
            self.num_frames,
            self.image_size,
            self.image_size,
        )
        return self.ddim_sample(shape, **kwargs)


def unnormalize_img(tensor):
    return tensor
