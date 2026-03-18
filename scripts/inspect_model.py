"""Inspect RVC model checkpoint to understand architecture."""
import torch

cpt = torch.load("models/da7ee7.pth", map_location="cpu", weights_only=False)
print("config:", cpt["config"])
print("sr:", cpt["sr"], "f0:", cpt["f0"], "version:", cpt["version"])
print()

weights = cpt["weight"]
# Group by prefix
groups = {}
for k in sorted(weights.keys()):
    prefix = k.split(".")[0]
    groups.setdefault(prefix, []).append(k)

for prefix, keys in sorted(groups.items()):
    print(f"\n--- {prefix} ({len(keys)} params) ---")
    for k in keys[:5]:
        print(f"  {k}: {weights[k].shape}")
    if len(keys) > 5:
        print(f"  ... ({len(keys) - 5} more)")

# Check config values
cfg = cpt["config"]
print(f"\nConfig interpretation:")
print(f"  filter_length: {cfg[0]}")
print(f"  segment_size: {cfg[1]}")
print(f"  inter_channels: {cfg[2]}")
print(f"  hidden_channels: {cfg[3]}")
print(f"  filter_channels: {cfg[4]}")
print(f"  n_heads: {cfg[5]}")
print(f"  n_layers: {cfg[6]}")
print(f"  kernel_size: {cfg[7]}")
print(f"  p_dropout: {cfg[8]}")
print(f"  resblock: {cfg[9]}")
print(f"  resblock_kernel_sizes: {cfg[10]}")
print(f"  resblock_dilation_sizes: {cfg[11]}")
print(f"  upsample_rates: {cfg[12]}")
print(f"  upsample_initial_channel: {cfg[13]}")
print(f"  upsample_kernel_sizes: {cfg[14]}")
print(f"  spk_embed_dim: {cfg[15]}")
print(f"  gin_channels: {cfg[16]}")
print(f"  sr: {cfg[17]}")
