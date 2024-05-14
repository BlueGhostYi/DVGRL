import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class DVGRL(nn.Module):
    def __init__(self, p_dims, q_dims=None, dataset=None, device=None, args=None, dropout=0.0):
        super(DVGRL, self).__init__()
        self.p_dims = p_dims
        self.q_dims = q_dims
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.device = device
        self.dataset = dataset
        self.args = args
        self.embedding_dim = args.dims
        
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        
        self.s_layers = nn.ModuleList([nn.Linear(self.dataset.num_users, self.embedding_dim * 2)])
 
        self.attention_dense = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False),
        )
    
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(p_dims[:-1], p_dims[1:])])
     
        self.s_p_layers = nn.ModuleList([nn.Linear(self.embedding_dim, self.dataset.num_users)])

        self.drop = nn.Dropout(dropout)

        self.Graph = self.dataset.bipartite_graph  # sparse matrix
        self.Graph = self.convert_sp_mat_to_sp_tensor(self.Graph)  # sparse tensor
        self.Graph = self.Graph.coalesce().to(self.device)  # Sort the edge index and remove redundancy
        
        self.S_Graph = self.dataset.social_graph  # sparse matrix
        self.S_Graph = self.convert_sp_mat_to_sp_tensor(self.S_Graph)  # sparse tensor
        self.S_Graph = self.S_Graph.coalesce().to(self.device)  # Sort the edge index and remove redundancy
        
        self.activation_layer = nn.Tanh()
        self.activation = nn.Sigmoid()

    def convert_sp_mat_to_sp_tensor(self, sp_mat):
        """
            coo.row: x in user-item graph
            coo.col: y in user-item graph
            coo.data: [value(x,y)]
        """
        coo = sp_mat.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        value = torch.FloatTensor(coo.data)

        sp_tensor = torch.sparse.FloatTensor(index, value, torch.Size(coo.shape))
        return sp_tensor

    def encode(self):
        for i, layer in enumerate(self.q_layers):
            if i == 0:
                h = layer(self.Graph)  # [U,I] * [I, 64] =[U, 64]
            else:
                h = layer(torch.sparse.mm(self.Graph.t(), h)) #[I, U] * [U, 64] = [I, 64]
            h = self.drop(h)

            if i != len(self.q_layers) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
                return mu, logvar
               
    def social_encode(self):
        for i, layer in enumerate(self.s_layers):
            if i == 0:
                h = layer(self.S_Graph)  
            else:
                h = layer(torch.sparse.mm(self.S_Graph, h))
            h = self.drop(h)

            if i != len(self.s_layers) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, :self.embedding_dim]
                logvar = h[:, self.embedding_dim:]
                return mu, logvar
       
    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
        return h

    def social_decode(self, z):
        h = z
        for i, layer in enumerate(self.s_p_layers):
            h = layer(h)
            if i != len(self.s_p_layers) - 1:
                h = torch.tanh(h)
        return h
       
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        if self.training:
            eps = torch.randn_like(std)
            z = eps.mul(std) + mu
        else: 
            z = mu + std
        return z

    def forward(self, inputs):
        mu, logvar = self.encode()       
        s_mu, s_logvar = self.social_encode()
        u_z = self.reparameterize(mu[inputs.long()], logvar[inputs.long()])
        s_z = self.reparameterize(s_mu[inputs.long()], s_logvar[inputs.long()])
        
        all_z =torch.cat([u_z, s_z], dim=1)
        score = self.attention_dense(all_z)

        z = score * u_z + (1 - score) * s_z
 
        recon_A = self.decode(z)
        recon_S = self.social_decode(s_z)
        
        return recon_A, recon_S, mu, logvar, s_mu, s_logvar, u_z, s_z

    def recon_loss(self, recon_x, x, mu, logvar, anneal=1.0):
        BCE = - torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))  # multi
        KLD = - 0.5 / recon_x.size(0) * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        return BCE + anneal * KLD

    def inter_recon_loss(self, recon_x, x, mu, logvar, s_mu, s_logvar, anneal=1.0):
        BCE = - torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))  # multi
        KLD = - 0.5 / recon_x.size(0) * torch.mean(torch.sum(1 + torch.log(logvar.exp()/s_logvar.exp() + 10e-6) - 
                                                             (mu - s_mu).pow(2)/(s_logvar.exp() + 10e-6) - logvar.exp()/(s_logvar.exp()+ 10e-6), dim=1))

        return BCE + anneal * KLD
