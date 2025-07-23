import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.nn.functional as F
import os
import cv2
import numpy as np
from PIL import Image

# Define the Encoder, Decoder, VectorQuantizer, and VQVAE classes


class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=224, z_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, hidden_dim, kernel_size=4, stride=2, padding=1
            ),  # 40x60
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1
            ),  # 20x30
            nn.ReLU(),
            nn.Conv2d(hidden_dim, z_dim, kernel_size=1),  # (B, z_dim, 20, 30)
        )

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, z_dim=64, hidden_dim=224, out_channels=3):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                z_dim, hidden_dim, kernel_size=4, stride=2, padding=1
            ),  # 40x60
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1
            ),  # 80x120
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.deconv(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z):
        # z: (B, C, H, W)
        z_perm = z.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        flat_z = z_perm.view(-1, self.embedding_dim)  # (B*H*W, C)

        distances = (
            torch.sum(flat_z**2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_z, self.embedding.weight.t())
            + torch.sum(self.embedding.weight**2, dim=1)
        )  # (B*H*W, num_embeddings)

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # (B*H*W, 1)
        encodings = torch.zeros(
            encoding_indices.size(0), self.num_embeddings, device=z.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight).view(z_perm.shape)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = z + (quantized - z).detach()

        return (
            quantized,
            loss,
            encoding_indices.view(z.shape[0], z.shape[2], z.shape[3]),
        )


class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.vq = VectorQuantizer()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, _ = self.vq(z)
        x_recon = self.decoder(z_q)
        recon_loss = F.mse_loss(x_recon, x)
        loss = recon_loss + vq_loss
        return x_recon, loss, recon_loss, vq_loss


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            img_array = np.array(img)
            images.append(img_array)
    return np.array(images)


def train():
    # Check if GPU is available and select the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    folder_path = "/home/mrinall/TEA/hsai-predictor/MonoLstm/version2/safety_detection_labeled_data"
    image_array = load_images_from_folder(folder_path)
    image_array = image_array[0:4000]

    model = VQVAE().to(device)  # Move the model to GPU if available
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 50

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x in image_array:
            x = x / 255.0
            x = torch.from_numpy(x).float()
            x = x.permute(2, 0, 1)  # Change to (C, H, W)
            x = x.unsqueeze(0)  # Add batch dimension
            x = x.to(device)  # Move data to the selected device

            optimizer.zero_grad()
            recon, total_loss, recon_loss, vq_loss = model(x)
            loss = total_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(image_array)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "vqv_vae_weights.pth")


def eval(x):
    model = VQVAE()
    checkpoint = torch.load("vqv_vae_weights.pth", weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()

    output, _, _, _ = model.forward(x)
    output = output.squeeze(0)
    output = output.permute(1, 2, 0)  # H, W, C
    output = output.detach().numpy()
    output = output[..., ::-1]  # RGB to BGR
    return output


if __name__ == "__main__":
    folder_path = (
        "/home/mrinall/TEA/hsai-predictor/MonoLstm/vqv/safety_detection_labeled_data"
    )
    # train()
    image_array = load_images_from_folder(folder_path)
    x = image_array[100]
    x = x / 255.0
    x = torch.from_numpy(x).float()
    x = x.permute(2, 0, 1)
    print(x.shape)
    x = x.unsqueeze(0)  # Batch

    out = eval(x)
    cv2.imshow("Image", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
