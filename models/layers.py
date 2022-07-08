import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, in_channels, embedding_dim=128, num_heads=4, num_layers=3):
        super(TransformerEncoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)  # takes shape S,n,E

        self.positional_encodings = nn.Parameter(torch.rand(500, embedding_dim), requires_grad=True)

    def forward(self, x):
        
        embeddings = x + self.positional_encodings[:x.shape[2], :].T.unsqueeze(0)

        # change to S,n,E format required by transformer
        embeddings = embeddings.permute(2, 0, 1)
        x = self.transformer_encoder(embeddings)  # .shape = S,n,E
        return x
