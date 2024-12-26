import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Encoder, Decoder

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, codebook_dim):
        super(VectorQuantizer, self).__init__()
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z_e):
        z_e_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, z_e.size(1))
        distances = torch.sum((z_e_flat.unsqueeze(1) - self.codebook.weight.unsqueeze(0)) ** 2, dim=2)
        indices = torch.argmin(distances, dim=1)
        z_q_flat = self.codebook(indices)
        z_q = z_q_flat.view(z_e.size()).permute(0, 3, 1, 2).contiguous()
        return z_q, indices

class VQVAELoss(nn.Module):
    def __init__(self, beta):
        super(VQVAELoss, self).__init__()
        self.beta = beta

    def forward(self, z_e, z_q):
        codebook_loss = F.mse_loss(z_e, z_q.detach())
        commitment_loss = F.mse_loss(z_e.detach(), z_q)
        total_loss = codebook_loss + self.beta * commitment_loss
        return total_loss, codebook_loss, commitment_loss
    
class Model(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(Model, self).__init__()
        
        self._encoder = Encoder(3, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)
        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(embedding_dim,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)
        self.vq_loss = VQVAELoss(0.25)

    def forward(self, x):
        z = self._encoder(x)

        z = self._pre_vq_conv(z)
        z_q, indices = self._vq_vae(z)
        total_loss, codebook_loss, commitment_loss = self.vq_loss(z, z_q)
        x_recon = self._decoder(z_q)

        return total_loss, x_recon
    

class VQVAE(nn.Module):
    def __init__(self, input_channels, hidden_dim, codebook_size, codebook_dim, beta):
        super(VQVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, codebook_dim, kernel_size=4, stride=2, padding=1)
        )
        self.vector_quantizer = VectorQuantizer(codebook_size, codebook_dim)
        self.vq_loss_fn = VQVAELoss(beta)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(codebook_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, input_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, _ = self.vector_quantizer(z_e)
        x_recon = self.decoder(z_q)

        vq_loss, codebook_loss, commitment_loss = self.vq_loss_fn(z_e, z_q)
        recon_loss = F.mse_loss(x_recon, x)
        total_loss = recon_loss + vq_loss

        return x_recon, total_loss, recon_loss, vq_loss, codebook_loss, commitment_loss

# CIFAR-10 Dataset Preparation
def get_cifar10_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

# Example usage
if __name__ == "__main__":
    input_channels = 3
    hidden_dim = 128
    codebook_size = 512
    codebook_dim = 64
    beta = 0.25
    batch_size = 64

    model = VQVAE(input_channels, hidden_dim, codebook_size, codebook_dim, beta)
    model = model.cuda() if torch.cuda.is_available() else model

    train_loader, test_loader = get_cifar10_dataloader(batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1):
        model.train()
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.cuda() if torch.cuda.is_available() else x

            optimizer.zero_grad()
            x_recon, total_loss, recon_loss, vq_loss, codebook_loss, commitment_loss = model(x)

            total_loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}: Total Loss = {total_loss.item()}, Recon Loss = {recon_loss.item()}, Codebook Loss = {codebook_loss.item()}, Commitment Loss = {commitment_loss.item()}")
