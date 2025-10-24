import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm, trange
import numpy as np
import random
import sys
import gc
from pympler import asizeof
from progress_table import ProgressTable
from DataLoader import RLDataProcessor, RLDataset


class DecisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embed_t = nn.Embedding(config.max_length, config.hidden_dim).to(torch.float16)
        self.embed_a = nn.Embedding(config.act_dim, config.hidden_dim).to(torch.float16)
        self.embed_r = nn.Linear(1, config.hidden_dim).to(torch.float16)

        self.embed_s = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4, padding=0).to(torch.float16),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0).to(torch.float16),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0).to(torch.float16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, config.hidden_dim).to(torch.float16),
            nn.Tanh()
        )

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=4 * config.hidden_dim,
            dropout=0.1
        ).to(torch.float16)

        self.transformer = nn.TransformerDecoder(
            self.decoder_layer,
            num_layers=config.n_layers
        ).to(torch.float16)

        self.pred_a = nn.Linear(config.hidden_dim, config.act_dim).to(torch.float16)

    def forward(self, s, a, r, t):
        s = s.to(torch.float16)
        r = r.to(torch.float16)

        pos_embedding = self.embed_t(t)
        a_embedding = self.embed_a(a) + pos_embedding
        r_embedding = self.embed_r(r.unsqueeze(-1)) + pos_embedding
        s = s.reshape(-1, 4, 84, 84)
        s_embedding = self.embed_s(s)
        s_embedding = s_embedding.view(2, 10, self.config.hidden_dim) + pos_embedding

        input_embeds = torch.stack((r_embedding, s_embedding, a_embedding), dim=2)
        input_embeds = input_embeds.flatten(1, 2)

        seq_len = input_embeds.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        input_embeds = input_embeds.permute(1, 0, 2)

        hidden_states = self.transformer(input_embeds, input_embeds, tgt_mask=causal_mask)
        hidden_states = hidden_states.permute(1, 0, 2)

        a_hidden = hidden_states[:, 2::3, :]

        return self.pred_a(a_hidden)


def train_model(model, dataloader, optimizer, criterion, n_epochs):
    ep_bar = trange(n_epochs, desc="epoch bar")
    model.train()
    batch_bar = None
    for epoch in ep_bar:
        batch_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=True)
        for s, a, r, t in batch_bar:
            # s, a, r, t = s.to("cuda"), a.to("cuda"), r.to("cuda"), t.to("cuda")
            optimizer.zero_grad()
            a_preds = model(s, a, r, t)
            a_preds = a_preds.view(-1, 6)
            a = a.view(-1)
            a = a.to(torch.long)
            loss = criterion(a_preds, a)
            print(loss)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_bar.set_postfix({"Loss": loss.item()})
        # ep_bar.set_description(f"Loss = {loss.item()}")


class Config():
    max_length = 2833
    act_dim = 6
    hidden_dim = 192
    n_heads = 3
    n_layers = 4

