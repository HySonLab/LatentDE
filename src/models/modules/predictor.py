import torch.nn as nn


class DropoutPredictor(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 hidden_dim: int = 128,
                 dropout: float = 0.2,
                 **kwargs):
        super(DropoutPredictor, self).__init__()
        self.input_layer = nn.Linear(latent_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = nn.functional.relu(self.dropout(self.input_layer(x)))
        x = self.output_layer(x)
        return x
