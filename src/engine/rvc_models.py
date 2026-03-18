"""
RVC v2 model architecture — SynthesizerTrnMs768NSFsid.

Self-contained implementation for inference only (no training components).
Supports standard RVC v2 checkpoints with 768-dim HuBERT features (v2)
and Neural Source Filter (NSF) HiFi-GAN decoder.

Reference: RVC-Project/Retrieval-based-Voice-Conversion-WebUI
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import remove_weight_norm


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Utility helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _fused_add_tanh_sigmoid_multiply(in_act: torch.Tensor, n_channels: int) -> torch.Tensor:
    t_act = torch.tanh(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    return t_act * s_act


def _init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        m.weight.data.normal_(mean, std)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Layer Norm (gamma/beta style used by RVC)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        return self.gamma.unsqueeze(-1) * x + self.beta.unsqueeze(-1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Multi-Head Attention with relative positional encoding
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MultiHeadAttention(nn.Module):
    def __init__(self, channels: int, out_channels: int, n_heads: int,
                 window_size: int = 10, p_dropout: float = 0.0):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        rel_std = self.k_channels ** -0.5
        self.emb_rel_k = nn.Parameter(
            torch.randn(1, window_size * 2 + 1, self.k_channels) * rel_std
        )
        self.emb_rel_v = nn.Parameter(
            torch.randn(1, window_size * 2 + 1, self.k_channels) * rel_std
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)
        x_out = self._attention(q, k, v, attn_mask)
        return self.conv_o(x_out)

    def _attention(self, query: torch.Tensor, key: torch.Tensor,
                   value: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        b, d, t_s = key.size()
        t_t = query.size(2)

        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))

        # Relative positional bias via index gather
        w = self.window_size
        # Build relative position index matrix: rel[i,j] = clamp(i - j + w, 0, 2w)
        idx_q = torch.arange(t_t, device=query.device).unsqueeze(1)
        idx_k = torch.arange(t_s, device=query.device).unsqueeze(0)
        rel_idx = (idx_q - idx_k + w).clamp(0, 2 * w)  # (t_t, t_s)

        # Gather relative key embeddings
        emb_k = self.emb_rel_k[0, rel_idx]  # (t_t, t_s, d_k)
        # query: (b, h, t_t, d_k) → compute dot product with rel embeddings
        rel_bias = torch.einsum("bhtd,tsd->bhts",
                                query / math.sqrt(self.k_channels), emb_k)
        scores = scores + rel_bias

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)

        output = torch.matmul(p_attn, value)

        # Relative positional values
        emb_v = self.emb_rel_v[0, rel_idx]  # (t_t, t_s, d_k)
        rel_val = torch.einsum("bhts,tsd->bhtd", p_attn, emb_v)
        output = output + rel_val

        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FFN (Feed-Forward Network)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FFN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, filter_channels: int,
                 kernel_size: int, p_dropout: float = 0.0):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size,
                                padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size,
                                padding=kernel_size // 2)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Encoder (Transformer-style)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Encoder(nn.Module):
    def __init__(self, hidden_channels: int, filter_channels: int, n_heads: int,
                 n_layers: int, kernel_size: int = 1, p_dropout: float = 0.0,
                 window_size: int = 10):
        super().__init__()
        self.n_layers = n_layers

        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for _ in range(n_layers):
            self.attn_layers.append(
                MultiHeadAttention(hidden_channels, hidden_channels, n_heads,
                                   window_size=window_size, p_dropout=p_dropout)
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(hidden_channels, hidden_channels, filter_channels,
                    kernel_size, p_dropout=p_dropout)
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        for i in range(self.n_layers):
            y = self.attn_layers[i](x * x_mask)
            y = F.dropout(y, training=self.training)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            y = F.dropout(y, training=self.training)
            x = self.norm_layers_2[i](x + y)
        return x * x_mask


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TextEncoder768 (for RVC v2 — 768-dim HuBERT features)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TextEncoder768(nn.Module):
    def __init__(self, out_channels: int, hidden_channels: int,
                 filter_channels: int, n_heads: int, n_layers: int,
                 kernel_size: int, p_dropout: float):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.emb_phone = nn.Linear(768, hidden_channels)
        self.emb_pitch = nn.Embedding(256, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        self.encoder = Encoder(
            hidden_channels, filter_channels, n_heads, n_layers,
            kernel_size, p_dropout,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, phone: torch.Tensor, pitch: torch.Tensor,
                lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            phone: (B, T, 768) HuBERT features
            pitch: (B, T) pitch indices (0-255)
            lengths: (B,) sequence lengths
        Returns:
            m, logs, x_mask
        """
        x = self.emb_phone(phone) + self.emb_pitch(pitch)
        x = x * math.sqrt(self.hidden_channels)
        x = self.lrelu(x)
        x = x.transpose(1, 2)  # (B, C, T)

        x_mask = torch.ones(x.size(0), 1, x.size(2), device=x.device, dtype=x.dtype)

        x = self.encoder(x, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = stats.split(self.out_channels, dim=1)
        return m, logs, x_mask


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  WaveNet (used in flow coupling layers)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class WN(nn.Module):
    def __init__(self, hidden_channels: int, kernel_size: int, dilation_rate: int,
                 n_layers: int, gin_channels: int = 0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()

        if gin_channels > 0:
            self.cond_layer = weight_norm(
                nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            )

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = (kernel_size * dilation - dilation) // 2
            self.in_layers.append(
                weight_norm(nn.Conv1d(hidden_channels, 2 * hidden_channels,
                                      kernel_size, dilation=dilation, padding=padding))
            )
            if i < n_layers - 1:
                res_skip_ch = 2 * hidden_channels
            else:
                res_skip_ch = hidden_channels
            self.res_skip_layers.append(
                weight_norm(nn.Conv1d(hidden_channels, res_skip_ch, 1))
            )

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor,
                g: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = torch.zeros_like(x)

        if g is not None and self.gin_channels > 0:
            g_all = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None and self.gin_channels > 0:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g_all[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
                x_in = x_in + g_l

            acts = _fused_add_tanh_sigmoid_multiply(x_in, self.hidden_channels)
            res_skip = self.res_skip_layers[i](acts)

            if i < self.n_layers - 1:
                x = (x + res_skip[:, :self.hidden_channels, :]) * x_mask
                output = output + res_skip[:, self.hidden_channels:, :]
            else:
                output = output + res_skip

        return output * x_mask


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ResidualCouplingLayer + Block (normalizing flow)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ResidualCouplingLayer(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, kernel_size: int,
                 dilation_rate: int, n_layers: int, gin_channels: int = 0,
                 mean_only: bool = False):
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers,
                      gin_channels=gin_channels)
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (1 if mean_only else 2), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor,
                g: Optional[torch.Tensor] = None, reverse: bool = False) -> torch.Tensor:
        x0, x1 = x.split(self.half_channels, dim=1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask

        if self.mean_only:
            m = stats
            logs = torch.zeros_like(m)
        else:
            m, logs = stats.split(self.half_channels, dim=1)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask

        return torch.cat([x0, x1], dim=1)


class Flip(nn.Module):
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return torch.flip(x, [1])


class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, kernel_size: int,
                 dilation_rate: int, n_layers: int, n_flows: int = 4,
                 gin_channels: int = 0):
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(channels, hidden_channels, kernel_size,
                                      dilation_rate, n_layers, gin_channels=gin_channels,
                                      mean_only=True)
            )
            self.flows.append(Flip())

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor,
                g: Optional[torch.Tensor] = None, reverse: bool = False) -> torch.Tensor:
        if not reverse:
            for flow in self.flows:
                x = flow(x, x_mask, g=g)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=True) if hasattr(flow, 'enc') else flow(x)
        return x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Neural Source Filter (NSF) source module
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SourceModuleHnNSF(nn.Module):
    """Generate harmonics + noise excitation signal from F0."""

    def __init__(self, sampling_rate: int, harmonic_num: int = 0,
                 sine_amp: float = 0.1, noise_std: float = 0.003,
                 voiced_threshold: float = 0.0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.sampling_rate = sampling_rate
        self.harmonic_num = harmonic_num

        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, f0: torch.Tensor, upp: int) -> torch.Tensor:
        """
        Args:
            f0: (B, T) fundamental frequency in Hz
            upp: upsampling factor (product of decoder upsample_rates)
        Returns:
            Source signal (B, 1, T*upp)
        """
        with torch.no_grad():
            f0_up = f0.unsqueeze(1).transpose(1, 2)  # (B, T, 1)
            f0_up = F.interpolate(f0_up.transpose(1, 2), scale_factor=float(upp),
                                   mode="nearest").transpose(1, 2)  # (B, T*upp, 1)

            # Generate sine wave from F0
            rad = f0_up / self.sampling_rate  # Normalize to [0, 1]

            # Cumulative phase
            phase = torch.cumsum(rad, dim=1) * 2 * math.pi
            sine_waves = torch.sin(phase) * self.sine_amp  # (B, T*upp, 1)

            # UV mask: 1 for voiced, 0 for unvoiced
            uv = (f0_up > 1.0).float()

            # Add harmonics
            if self.harmonic_num > 0:
                harmonics = []
                for k in range(1, self.harmonic_num + 1):
                    h_phase = torch.cumsum(rad * (k + 1), dim=1) * 2 * math.pi
                    harmonics.append(torch.sin(h_phase) * self.sine_amp)
                sine_waves = torch.cat([sine_waves] + harmonics, dim=-1)

            # Blend: voiced regions get sine, unvoiced get noise
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)
            sine_waves = sine_waves * uv + noise

        # Mix through learnable linear + tanh
        sine_waves = sine_waves.to(dtype=self.l_linear.weight.dtype)
        har_source = self.l_tanh(self.l_linear(sine_waves))  # (B, T*upp, 1)

        return har_source.transpose(1, 2)  # (B, 1, T*upp)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HiFi-GAN ResBlock
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3,
                 dilation: tuple = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for d in dilation:
            self.convs1.append(
                weight_norm(nn.Conv1d(channels, channels, kernel_size,
                                      dilation=d, padding=(kernel_size * d - d) // 2))
            )
            self.convs2.append(
                weight_norm(nn.Conv1d(channels, channels, kernel_size,
                                      dilation=1, padding=(kernel_size - 1) // 2))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GeneratorNSF (HiFi-GAN decoder with Neural Source Filter)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GeneratorNSF(nn.Module):
    def __init__(self, initial_channel: int, resblock_kernel_sizes: list[int],
                 resblock_dilation_sizes: list[list[int]],
                 upsample_rates: list[int], upsample_initial_channel: int,
                 upsample_kernel_sizes: list[int], gin_channels: int,
                 sr: int):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.upp = math.prod(upsample_rates)

        self.m_source = SourceModuleHnNSF(sampling_rate=sr)

        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, 7, padding=3)

        self.ups = nn.ModuleList()
        self.noise_convs = nn.ModuleList()

        ch = upsample_initial_channel
        stride_prod = math.prod(upsample_rates)

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(nn.ConvTranspose1d(
                    ch, ch // 2, k, stride=u, padding=(k - u) // 2
                ))
            )
            stride_prod //= u
            if stride_prod > 1:
                noise_k = stride_prod * 2
                noise_s = stride_prod
            else:
                noise_k = 1
                noise_s = 1
            self.noise_convs.append(
                nn.Conv1d(1, ch // 2, noise_k, stride=noise_s,
                          padding=(noise_k - noise_s) // 2 if noise_k > 1 else 0)
            )
            ch //= 2

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch_cur = upsample_initial_channel // (2 ** (i + 1))
            for k_size, d_sizes in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(ch_cur, k_size, tuple(d_sizes)))

        self.conv_post = nn.Conv1d(ch_cur, 1, 7, padding=3, bias=False)
        self.ups.apply(_init_weights)

        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x: torch.Tensor, f0: torch.Tensor,
                g: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) latent features
            f0: (B, T) fundamental frequency
            g: (B, gin_channels, 1) speaker embedding
        Returns:
            (B, 1, T * upp) waveform
        """
        har_source = self.m_source(f0, self.upp)  # (B, 1, T*upp)

        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)

            # Inject noise/source signal
            x_source = self.noise_convs[i](har_source)
            x = x + x_source[:, :, :x.size(2)]

            xs = None
            for j in range(self.num_kernels):
                rb = self.resblocks[i * self.num_kernels + j]
                if xs is None:
                    xs = rb(x)
                else:
                    xs = xs + rb(x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SynthesizerTrnMs768NSFsid  —  THE main RVC v2 model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SynthesizerTrnMs768NSFsid(nn.Module):
    """
    RVC v2 voice synthesis model.

    Inference forward pass:
        1. enc_p:  HuBERT features + pitch → (mu, logs)
        2. z = mu  (deterministic inference)
        3. flow (reverse): transform z using speaker-conditioned normalizing flow
        4. dec:  generate waveform from z + F0 + speaker embedding
    """

    def __init__(self, spec_channels: int, segment_size: int,
                 inter_channels: int, hidden_channels: int,
                 filter_channels: int, n_heads: int, n_layers: int,
                 kernel_size: int, p_dropout: float, resblock: str,
                 resblock_kernel_sizes: list[int],
                 resblock_dilation_sizes: list[list[int]],
                 upsample_rates: list[int],
                 upsample_initial_channel: int,
                 upsample_kernel_sizes: list[int],
                 spk_embed_dim: int, gin_channels: int, sr: int):
        super().__init__()

        self.segment_size = segment_size

        self.enc_p = TextEncoder768(
            inter_channels, hidden_channels, filter_channels,
            n_heads, n_layers, kernel_size, float(p_dropout),
        )

        self.dec = GeneratorNSF(
            inter_channels, resblock_kernel_sizes, resblock_dilation_sizes,
            upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
            gin_channels, sr,
        )

        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3,
            gin_channels=gin_channels,
        )

        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)

    def forward(self, phone: torch.Tensor, phone_lengths: torch.Tensor,
                pitch: torch.Tensor, pitchf: torch.Tensor,
                sid: torch.Tensor) -> torch.Tensor:
        """
        Inference forward.

        Args:
            phone:        (B, T, 768) HuBERT features
            phone_lengths: (B,) sequence lengths
            pitch:         (B, T) quantized pitch (0-255)
            pitchf:        (B, T) continuous F0 in Hz
            sid:           (B,) speaker ID

        Returns:
            (B, 1, T_audio) generated waveform
        """
        g = self.emb_g(sid).unsqueeze(-1)  # (B, gin_channels, 1)

        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)

        # Sample z_p from prior (mu + noise * sigma * 0.66666)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask

        # Reverse flow
        z = self.flow(z_p, x_mask, g=g, reverse=True)

        # Decode to audio
        o = self.dec(z * x_mask, pitchf, g=g)
        return o


def build_model_from_checkpoint(cpt: dict, device: str = "cpu") -> SynthesizerTrnMs768NSFsid:
    """
    Build a SynthesizerTrnMs768NSFsid model from an RVC v2 checkpoint dict.

    Args:
        cpt: Loaded torch checkpoint dict (must have 'config' and 'weight' keys).
        device: Target device ("cpu" or "cuda").

    Returns:
        Loaded model in eval mode.
    """
    config = cpt["config"]

    model = SynthesizerTrnMs768NSFsid(
        spec_channels=config[0],
        segment_size=config[1],
        inter_channels=config[2],
        hidden_channels=config[3],
        filter_channels=config[4],
        n_heads=config[5],
        n_layers=config[6],
        kernel_size=config[7],
        p_dropout=config[8],
        resblock=config[9],
        resblock_kernel_sizes=config[10],
        resblock_dilation_sizes=config[11],
        upsample_rates=config[12],
        upsample_initial_channel=config[13],
        upsample_kernel_sizes=config[14],
        spk_embed_dim=config[15],
        gin_channels=config[16],
        sr=config[17],
    )

    # Load weights (strict=False tolerates missing posterior_encoder, etc.)
    model.load_state_dict(cpt["weight"], strict=False)
    model.eval()
    model.to(device)
    return model
