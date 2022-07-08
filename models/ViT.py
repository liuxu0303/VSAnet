import torch
import torch.nn as nn

from .layers import TransformerEncoder


class ViT(nn.Module):
    def __init__(self, in_channels, n_query_channels=128, dim_out=256, 
                 embedding_dim=128, num_heads=4, num_layers=4, norm='linear'):
        super(ViT, self).__init__()
        self.norm = norm
        self.n_query_channels = n_query_channels
        self.transformer = TransformerEncoder(in_channels, embedding_dim, num_heads, num_layers)

        self.regressor0 = nn.Sequential(nn.Linear(embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out))
        self.regressor1 = nn.Sequential(nn.Linear(embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out),
                                       nn.Softmax(dim=1))

    def forward(self, x):
     
        tgt = self.transformer(x.clone())

        first_output_embedding, second_output_embedding = tgt[0, ...], tgt[1, ...]


        areabin_weights = self.regressor0(first_output_embedding)  # .shape = n, dim_out
        softmax_p = self.regressor1(second_output_embedding) # .shape = n, N
        if self.norm == 'linear':
            areabin_weights = torch.relu(areabin_weights)
            eps = 0.1
            areabin_weights = areabin_weights + eps
        elif self.norm == 'softmax':
            return torch.softmax(areabin_weights, dim=1), softmax_p
        else:
            areabin_weights = torch.sigmoid(areabin_weights)
        areabin_widths_normed = areabin_weights / areabin_weights.sum(dim=1, keepdim=True)
        return areabin_widths_normed, softmax_p
