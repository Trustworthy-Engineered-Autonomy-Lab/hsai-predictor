import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

# pick the right device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE(nn.Module):
    def __init__(self, latent_size=32):
        super().__init__()
        self.latent_size = latent_size
        # sum reconstruction loss
        self.MSE = nn.MSELoss(reduction="sum")

        # Encoder: (3,64,64) -> (256,4,4)
        self.enc_conv1 = nn.Conv2d(3, 32, 4, 2, 1)
        self.enc_conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.enc_conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.enc_conv4 = nn.Conv2d(128, 256, 4, 2, 1)
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_size)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_size)

        # Decoder: latent -> (256,4,4) -> (3,64,64)
        self.dec_fc = nn.Linear(latent_size, 256 * 4 * 4)
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.dec_conv4 = nn.ConvTranspose2d(32, 3, 4, 2, 1)

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = x.flatten(1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        # clamp logvar to avoid extreme KL
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.dec_fc(z)).view(-1, 256, 4, 4)
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = F.relu(self.dec_conv3(x))
        return torch.sigmoid(self.dec_conv4(x))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def forward_eval(self, x):
        mu, _ = self.encode(x)
        return self.decode(mu)

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        # reconstruction: sum over pixels (batch=1)
        rec_loss = self.MSE(recon_x, x)
        # KL divergence: sum dims
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return rec_loss + beta * kl_loss, rec_loss, kl_loss

    def train_step(self, x, optimizer, beta):
        self.train()
        x = x.to(device)
        recon, mu, logvar = self.forward(x)
        loss, rec, kl = self.loss_function(recon, x, mu, logvar, beta)
        optimizer.zero_grad()
        loss.backward()
        # clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item(), rec.item(), kl.item()


def train():
    from data_loader import load_images

    images = load_images()
    model = VAE(latent_size=32).to(device)
    # reduced LR to stabilize
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 50
    warmup_epochs = 200

    for epoch in range(num_epochs):
        beta = min(1.0, epoch / warmup_epochs)
        total_l = total_r = total_k = 0.0
        for img in images:
            img = img.unsqueeze(0)
            l, r, k = model.train_step(img, optimizer, beta)
            total_l += l
            total_r += r
            total_k += k

        n = len(images)
        print(
            f"Epoch {epoch + 1}/{num_epochs} β={beta:.2f} Loss={total_l / n:.3f} Rec={total_r / n:.3f} KL={total_k / n:.3f}"
        )
        torch.save(model.state_dict(), "./weights/vae_weights_split.pth")


def eval(x):
    model = VAE(latent_size=32).to(device)
    model.load_state_dict(
        torch.load("./weights/vae_weights_split.pth", map_location=device)
    )
    model.eval()
    with torch.no_grad():
        out = model.forward_eval(x.to(device))
    out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out = (out * 255).clip(0, 255).astype(np.uint8)[..., ::-1]
    return out


if __name__ == "__main__":
    print(f"Using device: {device}")
    # train()

    from data_loader import load_images

    imgs = load_images()
    x = imgs[0].unsqueeze(0)
    inp = (x[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
    cv2.imshow("Input", inp)
    cv2.waitKey(0)
    out = eval(x)
    cv2.imshow("Recon", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
