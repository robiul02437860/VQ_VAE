import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from ResNet_model import Encoder, Decoder
from Vector_Qunatizer import VectorQuantizer, VQVAELoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 256
# num_training_updates = 15000
n_epochs = 500
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2
embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25
decay = 0.99
learning_rate = 1e-3

class  Model(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self._encoder = Encoder(in_channels, num_hiddens, num_residual_layers, num_residual_hiddens)
        self.pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)
        self._vq = VectorQuantizer(num_embeddings, embedding_dim)
        self._decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)
        self.vq_loss = VQVAELoss(beta=beta)
    

    def forward(self, x):
        x = self._encoder(x)
        # print("encoder", x.shape)
        z_e = self.pre_vq_conv(x)
        # print("ze shape", z_e.shape)
        z_q = self._vq(z_e)
        # print("z_q shape", z_q.shape)
        total_loss, codebook_loss, commitment_loss = self.vq_loss(z_e, z_q)
        x_recons = self._decoder(z_q)
        return x_recons, total_loss

if __name__=='__main__':
    training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))
    validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))
    training_loader = DataLoader(training_data, 
                             batch_size=batch_size, 
                             shuffle=True,
                             pin_memory=True)
    validation_loader = DataLoader(validation_data,
                               batch_size=32,
                               shuffle=True,
                               pin_memory=True)
    
    
    model = Model(3, num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim).to(device)
    # x = torch.rand(1, 3, 32, 32)
    # model(x)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for i in range(n_epochs):
        model.train()
        for idx, (batch_data, _) in enumerate(training_loader):
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            x_recons, vq_loss = model(batch_data)
            recon_error = F.mse_loss(x_recons, batch_data)
            loss = recon_error + vq_loss
            loss.backward()

            optimizer.step()

            if (idx+1) % 100 == 0:
                print('%d idx' % (idx+1))
                print('recon_error: %.3f' % recon_error.item())
                print('total loss: %.3f' % loss.item())
                print()
