from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from vae import VAE  # your existing encoder class


def load_embed_label_list(
    split: str = "train",
    folder: str | Path = "../cartpoledata",
    weights: str | Path = "./weights/vae_weights_split.pth",
    latent_size: int = 32,
    device: str = "cpu",
) -> List[Dict[str, torch.Tensor | bool]]:
    """Returns embeddings and labels inferred from subfolders 'safe' and 'unsafe' in the given split."""
    # Initialize VAE
    vae = VAE(latent_size=latent_size).to(device).eval()
    vae.load_state_dict(torch.load(weights, map_location=device))

    # Prepare data directory
    data_dir = Path(folder).expanduser().resolve() / split
    if not data_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {data_dir}")

    records: List[Dict[str, torch.Tensor | bool]] = []
    # Iterate over classes
    for label_name in ["safe", "unsafe"]:
        class_dir = data_dir / label_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")
        label_value = True if label_name == "safe" else False

        # Process each image file
        for img_path in class_dir.iterdir():
            if not img_path.is_file() or img_path.suffix.lower() not in [
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
            ]:
                continue
            # Load and preprocess image
            image = Image.open(img_path).convert("RGB")
            x = torch.from_numpy(np.array(image)).float() / 255.0  # [H,W,C]
            x = x.permute(2, 0, 1).unsqueeze(0).to(device)  # [1,C,H,W]

            # Encode
            z, _ = vae.encode(x=x)  # [1, latent_size]
            records.append({
                "embedding": z.squeeze(0).cpu(),
                "label": label_value,
            })

    return records


def load_images(
    split: str = "train",
    folder: str | Path = "../cartpoledata",
    device: str = "cpu",
) -> List[torch.Tensor]:
    """Returns a list of preprocessed image tensors from 'safe' and 'unsafe' folders for the given split."""
    data_dir = Path(folder).expanduser().resolve() / split
    if not data_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {data_dir}")

    images_list: List[torch.Tensor] = []
    for label_name in ["safe", "unsafe"]:
        class_dir = data_dir / label_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")

        for img_path in class_dir.iterdir():
            if not img_path.is_file() or img_path.suffix.lower() not in [
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
            ]:
                continue
            # Load and preprocess image
            image = Image.open(img_path).convert("RGB")
            x = torch.from_numpy(np.array(image)).float() / 255.0  # [H,W,C]
            x = x.permute(2, 0, 1).to(device)  # [C,H,W]

            images_list.append(x)

    return images_list


# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage
    train_embeds = load_embed_label_list(split="train")
    print(f"Train embedding samples: {len(train_embeds)}")
    test_embeds = load_embed_label_list(split="test")
    print(f"Test embedding samples: {len(test_embeds)}")

    train_imgs = load_images(split="train")
    print(f"Train image tensors: {len(train_imgs)}")
    test_imgs = load_images(split="test")
    print(f"Test image tensors: {len(test_imgs)}")
