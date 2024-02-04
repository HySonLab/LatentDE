import os
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel
from typing import List, Union
from .decoder import Decoder


class ESM1b_Attention1d(nn.Module):

    def __init__(self):
        super(ESM1b_Attention1d, self).__init__()
        self.encoder = EsmModel.from_pretrained("facebook/esm1b_t33_650M_UR50S")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")
        self.decoder = Decoder()

    def forward(self, inputs):
        x = self.encoder(**inputs).last_hidden_state
        x = self.decoder(x)
        return x


class ESM1b_Landscape:
    """
        An ESM-based oracle model to simulate protein fitness landscape.
    """

    def __init__(self, task: str, device: Union[str, torch.device]):
        task_dir_path = os.path.join('./landscape_params/esm1b_landscape', task)
        task_dir_path = os.path.abspath(task_dir_path)
        assert os.path.exists(os.path.join(task_dir_path, 'decoder.pt'))
        self.model = ESM1b_Attention1d()
        self.model.decoder.load_state_dict(
            torch.load(os.path.join(task_dir_path, 'decoder.pt'))
        )
        with open(os.path.join(task_dir_path, 'starting_sequence.json')) as f:
            self.starting_sequence = json.load(f)

        self.tokenizer = self.model.tokenizer
        self.device = device
        self.model.to(self.device)

    def infer_fitness(self, sequences: List[str], device=None):
        # Input:  - sequences:      [query_batch_size, sequence_length]
        # Output: - fitness_scores: [query_batch_size]

        self.model.eval()
        fitness_scores = []
        for seq in sequences:
            inputs = self.tokenizer(seq, return_tensors="pt").to(self.device)
            fitness_scores.append(self.model(inputs).item())
        return fitness_scores
