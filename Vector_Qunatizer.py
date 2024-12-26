import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embedding, embedding_dim):
        super().__init__()
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embedding_dim)
        self.embedding.weight.data.uniform_(-1.0/num_embedding, 1.0/num_embedding)
    
    def forward(self, z_e):
        z_e_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, z_e.size(1))
        distances = torch.sum((z_e_flat.unsqueeze(1)-self.embedding.weight.unsqueeze(0))**2, dim=2)
        indices = torch.argmin(distances, dim=1)
        z_q_flat = self.embedding(indices)
        z_q = z_q_flat.view(-1, z_e.size(2), z_e.size(3), z_e.size(1)).permute(0, 3, 1, 2).contiguous()
        return z_q

class VQVAELoss(nn.Module):
    def __init__(self, beta):
        super(VQVAELoss, self).__init__()
        self.beta = beta

    def forward(self, z_e, z_q):
        codebook_loss = F.mse_loss(z_e, z_q.detach())
        commitment_loss = F.mse_loss(z_e.detach(), z_q)
        total_loss = codebook_loss + self.beta * commitment_loss
        return total_loss, codebook_loss, commitment_loss






