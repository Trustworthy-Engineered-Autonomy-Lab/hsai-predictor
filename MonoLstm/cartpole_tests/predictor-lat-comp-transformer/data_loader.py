# ─── embed_label_loader.py ───────────────────────────────────────────────
import math
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from vae import VAE  # your existing encoder class


def _load_npz(file_number: int, folder: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (images, states) from <folder>/<file_number>.npz
    ──► images  -> data.files[0]
    ──► states  -> data.files[2]   ← fixed (was [1])
    """
    p = Path(folder).expanduser().resolve() / f"{file_number}.npz"
    if not p.exists():
        raise FileNotFoundError(p)

    with np.load(p) as d:
        images = d[d.files[0]]
        states = d[d.files[2]]  # ←  correct index
    return images, states


@torch.no_grad()
def load_embed_label_list(
    file_number: int,
    folder: str | Path = "../small_cartpole",
    weights: str | Path = "./weights/vae_weights_split.pth",
    latent_size: int = 32,
    device: str = "cpu",
) -> List[Dict[str, torch.Tensor | bool]]:
    """List → records[index]['embedding'] / ['label'] works."""
    vae = VAE(latent_size=latent_size).to(device).eval()
    vae.load_state_dict(torch.load(weights, map_location=device))

    imgs, states = _load_npz(file_number, folder)

    records: List[Dict[str, torch.Tensor | bool]] = []
    for img, state in zip(imgs, states):
        x = torch.from_numpy(img).float() / 255.0  # [H,W,C]
        x = x.permute(2, 0, 1).unsqueeze(0).to(device)  # [1,C,H,W]
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, (224, 224), mode="bilinear", align_corners=False)

        z, _ = vae.encode(x=x)  # [1, latent_size]

        pole_angle_rad = state[2]  # 3rd component
        label = abs(math.degrees(pole_angle_rad)) < 45

        records.append({
            "embedding": z.squeeze(0).cpu(),  # Tensor[latent_size]
            "label": label,
        })

    return records


# -----------------------------------------------------------------------
if __name__ == "__main__":
    data = load_embed_label_list(1)  # loads 17.npz
    print(f"frames in file 100: {len(data)}")
    print("first embedding shape:", data[0]["embedding"].shape)
    print("first label:", data[0]["label"])
