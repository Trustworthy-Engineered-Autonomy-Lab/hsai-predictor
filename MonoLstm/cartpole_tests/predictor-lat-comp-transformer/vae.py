import math
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from PIL import Image
from matplotlib.widgets import Slider


device = "cuda"


class VAE(nn.Module):
    def __init__(self, latent_size=32):
        super(VAE, self).__init__()
        self.latent_size = latent_size

        # ---------------------------
        #         Encoder
        # ---------------------------
        # Input shape: (3, 224, 224)
        # Downsample by factor of 2 with each conv layer:
        #   224 -> 112 -> 56 -> 28 -> 14
        self.enc_conv1 = nn.Conv2d(
            3, 32, kernel_size=4, stride=2, padding=1
        )  # (3, 224,224) -> (32, 112,112)
        self.enc_conv2 = nn.Conv2d(
            32, 64, kernel_size=4, stride=2, padding=1
        )  # (32,112,112) -> (64, 56,56)
        self.enc_conv3 = nn.Conv2d(
            64, 128, kernel_size=4, stride=2, padding=1
        )  # (64, 56,56)  -> (128,28,28)
        self.enc_conv4 = nn.Conv2d(
            128, 256, kernel_size=4, stride=2, padding=1
        )  # (128,28,28)  -> (256,14,14)

        # Now the flattened feature map is (256 * 14 * 14) = 50176
        self.fc_mu = nn.Linear(256 * 14 * 14, latent_size)
        self.fc_logvar = nn.Linear(256 * 14 * 14, latent_size)

        # ---------------------------
        #         Decoder
        # ---------------------------
        # We do the reverse: 14 -> 28 -> 56 -> 112 -> 224
        self.dec_fc = nn.Linear(latent_size, 256 * 14 * 14)

        self.dec_conv1 = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1
        )  # (256,14,14) -> (128,28,28)
        self.dec_conv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )  # (128,28,28)-> (64,56,56)
        self.dec_conv3 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1
        )  # (64,56,56) -> (32,112,112)
        self.dec_conv4 = nn.ConvTranspose2d(
            32, 3, kernel_size=4, stride=2, padding=1
        )  # (32,112,112)-> (3,224,224)

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        x = F.relu(self.dec_fc(z))
        x = x.view(-1, 256, 14, 14)
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = F.relu(self.dec_conv3(x))
        x = F.relu(self.dec_conv4(x))
        return x

    def reparameterize_mean(self, mu, logvar):
        # Use mean value for reparameterization
        return mu

    def reparameterize(self, mu, logvar):
        # Reparameterization trick to sample from N(mu, var) from N(0,1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def train_step(self, x, optimizer):
        # Set the model to training mode
        self.train()
        # Forward pass
        if x.shape[2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        x = x.to(device)
        recon_x, mu, logvar = self.forward(x)
        # Compute loss
        loss = self.loss_function(recon_x, x, mu, logvar)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    # Define the loss function for the VAE
    def loss_function(self, recon_x, x, mu, logvar):
        MSE = nn.MSELoss(reduction="sum")
        reconstruction_loss = MSE(recon_x, x)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + kl_divergence

    def forward(self, x):
        # Forward pass through the VAE
        # Resize to (96, 144) if necessary
        if x.shape[2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def forward_eval(self, x):
        # Forward pass through the VAE
        # Resize to (96, 144) if necessary
        if x.shape[2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        x = x.to(device)
        mu, logvar = self.encode(x)
        z = self.reparameterize_mean(mu, logvar)
        return self.decode(z)


def train():
    from data_loader import load_images

    image_array = load_images()
    model = VAE(latent_size=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 50
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x in image_array:
            x = x / 255.0
            x = torch.from_numpy(x).float()
            x = x.permute(2, 0, 1)
            x = x.unsqueeze(0)  # Batch
            loss = model.train_step(x, optimizer)
            epoch_loss += loss
            # print(f"Loss: {loss}")

        avg_loss = epoch_loss / len(image_array)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), "./weights/vae_weights_split.pth")


def eval(x):
    model = VAE(latent_size=32).to(device)
    checkpoint = torch.load("./weights/vae_weights_split.pth", weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()
    if x.shape[2:] != (224, 224):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

    print(x.shape)
    output = model.forward_eval(x)
    output = output.squeeze(0)
    output = output.permute(1, 2, 0)  # H, W, C
    output = output.cpu().detach().numpy()
    output = output[..., ::-1]  # RGB to BGR
    return output


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # train()

    from data_loader import load_images

    image_array = load_images()
    x = image_array[0]
    x = x / 255.0
    x = torch.from_numpy(x).float()
    x = x.permute(2, 0, 1)
    x = x.unsqueeze(0)  # Batch
    if x.shape[2:] != (224, 224):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

    img = x[0].detach().cpu().numpy()  # shape: (C, H, W)
    img = np.transpose(img, (1, 2, 0))  # shape: (H, W, C)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)  # Scale to [0, 255]

    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Display image
    cv2.imshow("Image", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    out = eval(x)
    cv2.imshow("Image", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
