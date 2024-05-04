import os
import numpy as np
import json
import argparse

import config
from GPT import GPT
from StimulusModel import LMFeatures
from utils_stim import get_stim
from utils_resp import get_resp
from utils_ridge.ridge import ridge, bootstrap_ridge

import torch, tqdm
import matplotlib.pyplot as plt
np.random.seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FMRIDataset(torch.utils.data.Dataset):
    def __init__(self, stim, resp):
        self.stim = torch.tensor(stim, dtype=torch.float32)
        self.resp = torch.tensor(resp, dtype=torch.float32)

    def __len__(self):
        return self.stim.shape[0]

    def __getitem__(self, index):
        return self.stim[index], self.resp[index]


def calc_loss(y_pred, y_true, loss_fn):
    # mean over mse errors of each voxel. finally, mean over batch.
    loss = loss_fn(y_pred, y_true)
    loss = torch.mean(loss, dim=-1)
    loss = torch.mean(loss)
    return loss


class LinearModel(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=d_in, out_features=d_out)
        self.activation = torch.nn.LeakyReLU()

    def forward(self, X):
        return self.activation(self.linear(X))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--gpt", type = str, default = "perceived")
    parser.add_argument("--sessions", nargs = "+", type = int, default = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    args = parser.parse_args("--subject S1".split())

    # training stories
    stories = []
    with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f) 
    for sess in args.sessions:
        stories.extend(sess_to_story[str(sess)])

    # load gpt
    with open(os.path.join(config.DATA_LM_DIR, args.gpt, "vocab.json"), "r") as f:
        gpt_vocab = json.load(f)
    gpt = GPT(path = os.path.join(config.DATA_LM_DIR, args.gpt, "model"), vocab = gpt_vocab, device = config.GPT_DEVICE)
    features = LMFeatures(model = gpt, layer = config.GPT_LAYER, context_words = config.GPT_WORDS)

    # estimate encoding model
    rstim, tr_stats, word_stats = get_stim(stories, features)
    rresp = get_resp(args.subject, stories, stack = True)
    nchunks = int(np.ceil(rresp.shape[0] / 5 / config.CHUNKLEN))

    train_dataset = FMRIDataset(rstim, rresp)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)

    model = LinearModel(3072, 81126).to(device)
    mse_loss_fn = torch.nn.MSELoss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    n_epochs = 10
    losses = []
    for i in range(n_epochs):
        batch_losses = []
        model.train()
        for data in tqdm.tqdm(train_loader):
            X, y_true = data[0].to(device), data[1].to(device)
            y_pred = model(X)
            loss = calc_loss(y_pred, y_true, mse_loss_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        losses.append(np.mean(batch_losses))

    print(losses)