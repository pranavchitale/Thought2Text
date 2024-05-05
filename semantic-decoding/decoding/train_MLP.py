import os
import numpy as np
import json
import argparse
import torch, tqdm
from matplotlib import pyplot as plt

import config
from GPT import GPT
from StimulusModel import LMFeatures
from utils_stim import get_stim
from utils_resp import get_resp
from utils_ridge.ridge import ridge, bootstrap_ridge

# Experiments:
# CUDA_VISIBLE_DEVICES=1,2,3 python decoding/train_MLP.py --num_epochs 20 --batch_size 256 --lr 1e-5 --wd 1e-5 --save_name mlp_perceived_20_1e-5_1e-5.pth > exps/mlp_20_1e-5_1e-5.txt
# CUDA_VISIBLE_DEVICES=1,2,3 python decoding/train_MLP.py --num_epochs 20 --batch_size 256 --lr 1e-3 --wd 1e-5 --save_name mlp_perceived_20_1e-3_1e-5.pth > exps/mlp_20_1e-3_1e-5.txt
# CUDA_VISIBLE_DEVICES=1,2,3 python decoding/train_MLP.py --num_epochs 20 --batch_size 256 --lr 1e-2 --wd 1e-5 --save_name mlp_perceived_20_1e-2_1e-5.pth > exps/mlp_20_1e-2_1e-5.txt

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
np.random.seed(42)


class FMRIDataset(torch.utils.data.Dataset):
    def __init__(self, stim, resp):
        self.stim = torch.from_numpy(stim).float()
        self.resp = torch.from_numpy(resp).float()

    def __len__(self):
        return self.stim.shape[0]

    def __getitem__(self, index):
        return self.stim[index], self.resp[index]


class MLP(torch.nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(d_in, d_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(d_in, d_out)
        )

    def forward(self, x):
        return self.layers(x)


def calc_loss(presp, rresp):
    # mean over mse errors of each voxel. finally, mean over batch.
    loss = torch.square(presp - rresp)
    loss  = torch.mean(loss, dim=1)
    loss = torch.mean(loss)
    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, default = "S1")
    parser.add_argument("--gpt", type = str, default = "perceived")
    parser.add_argument("--sessions", nargs = "+", type = int, default = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    parser.add_argument("--num_epochs", type = int, default = 10)
    parser.add_argument("--batch_size", type = int, default = 64)
    parser.add_argument("--lr", type = float, default = 0.01)
    parser.add_argument("--wd", type = float, default = 0.1)
    parser.add_argument("--load_path", type = str, default="", help="specify filepath to pretrained model to resume training")
    parser.add_argument("--save_name", type = str, default="", help="specify a filename to save model after training. if empty, it will not save...\t[e.g. `mlp_perceived_01.pth`]")
    args = parser.parse_args()

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

    # load stimulus + responses data
    rstim, tr_stats, word_stats = get_stim(stories, features)
    rresp = get_resp(args.subject, stories, stack = True)
    train_dataset = FMRIDataset(rstim, rresp)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    model = MLP(3072, 3072, 81126).to(DEVICE)
    if args.load_path:
        model_state_dict = torch.load(args.load_path, map_location=DEVICE)
        if "module." in list(model_state_dict.keys())[0]:
            model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
        model.load_state_dict(model_state_dict)
        print('Resuming model training from checkpoint...')

    # DataParallel multi-GPU training (if available)
    if torch.cuda.is_available():
        devices = list(range(torch.cuda.device_count()))
        if len(devices) > 1:
            model = torch.nn.DataParallel(model, device_ids=devices)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    print('Starting Training Loop...')
    model.train()
    for epoch in range(args.num_epochs):
        batch_losses = []
        batches = tqdm.tqdm(train_loader)
        for idx, data in enumerate(batches):
            rstim, rresp = data[0].to(DEVICE), data[1].to(DEVICE)
            presp = model(rstim)
            loss = calc_loss(presp, rresp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            print(f'\tBatch: {idx + 1}/{len(batches)}\tBatch Loss: {batch_losses[-1]}')
        print(f'Epoch {epoch + 1}/{args.num_epochs}\tAverage Loss: {np.mean(batch_losses)}\t')

    # Save Model
    if args.save_name:
        save_location = os.path.join(config.MODEL_DIR, args.subject)
        os.makedirs(save_location, exist_ok = True)
        torch.save(model.state_dict(), os.path.join(save_location, args.save_name))
    

