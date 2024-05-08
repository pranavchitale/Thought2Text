"""
Filename: eval_encoders.py
Author(s): 
- Rajath Rao (rajath.rao@stonybrook.edu)
- Pranav Chitale (pranavshailesh.chitale@stonybrook.edu)
- Ashutosh Tiwari (ashutosh.tiwari@stonybrook.edu)

Usage:
$ python semantic-decoding/decoding/eval_encoders.py --model BASE --load_path semantic-decoding/models/S1/encoding_model_perceived.npz
$ python semantic-decoding/decoding/eval_encoders.py --model MLP --load_path semantic-decoding/models/S1/mlp_perceived_1e-3_1e-5.pth

System Requirements:
- Operating System: Ubuntu
- Python Version: Python 3.10.*
- Dependencies: (conda) environment.yaml

Description:
This file evaluates different variations of the EncoderModel. The options are BASE which is the baseline encoder, MLP which is with the
multi-layer perceptron instead of bootstrapped ridge regressions, and GPT2 which is the `distilgpt2` extracted stimulus feeding to the EM.
This script loads the specified pretrained encoder and evaluates it with accuracy metrics (I. Syntax | Classification) beginning from Line 77.
The metrics include mean-squared-error, residuals-squared which are both voxel-wise measures in this case. The encoder weights can come from
a regularized bootstrapped linear regression with Ridge L2 (I. Syntax | Classification) on Line 85.
"""


import os
import numpy as np
import json
import argparse
import torch

import config
from train_MLP import FMRIDataset, MLP, calc_loss
from GPT import GPT
from StimulusModel import LMFeatures
from utils_stim import get_stim
from utils_resp import get_resp


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
np.random.seed(42)

# CUDA_VISIBLE_DEVICES=1 python semantic-decoding/decoding/eval_encoders.py --model BASE --load_path semantic-decoding/models/S1/encoding_model_perceived.npz
#   BASE (voxel-wise MSE): 0.9394
# CUDA_VISIBLE_DEVICES=1 python semantic-decoding/decoding/eval_encoders.py --model MLP --load_path semantic-decoding/models/S1/mlp_perceived_1e-3_1e-5.pth
#   MLP (voxel-wise MSE): 0.7740

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, default = "S1")
    parser.add_argument("--gpt", type = str, default = "perceived", choices=["perceived", "imagined"])
    parser.add_argument("--sessions", nargs = "+", type = int, 
        default = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    parser.add_argument("--model", type = str, default = "BASE", choices=["BASE", "MLP", "GPT2"])
    parser.add_argument("--load_path", type = str, required=True, help="Specify path to load checkpoint parameters")
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
    
    # Prepare stimulus + response data
    em_cp = np.load("semantic-decoding/models/S1/encoding_model_perceived.npz")
    rstim, tr_stats, word_stats = get_stim(stories, features)
    rresp = get_resp(args.subject, stories, stack = True)[:, em_cp["voxels"]]
    train_dataset = FMRIDataset(rstim, rresp)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)

    print(f"Beginning Evaluation with {args.model}")
    if args.model == "BASE":
        rstim = torch.from_numpy(rstim).float().to(DEVICE)
        rresp = torch.from_numpy(rresp).float().to(DEVICE)

        # Load weights
        weights = torch.from_numpy(em_cp["weights"])[:, em_cp["voxels"]].float().to(DEVICE)

        # get response predictions
        presp = torch.matmul(rstim, weights)

        # calculate error
        mse = calc_loss(presp, rresp)
        print("Overall MSE (voxel-wise):", mse.item())

    elif args.model == "MLP":
        model = MLP(3072, config.VOXELS).to(DEVICE)

        # Load weights
        model_state_dict = torch.load(args.load_path, map_location=DEVICE)
        if "module." in list(model_state_dict.keys())[0]:
            model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
        model.load_state_dict(model_state_dict)

        residuals = []
        model.eval()
        for idx, batch in enumerate(train_loader):
            rstim, rresp = batch[0].to(DEVICE), batch[1].to(DEVICE)
            presp = model(rstim)
            
            # calculate error
            res = torch.square(presp - rresp)
            res = torch.mean(res, dim=1)
            residuals.append(res)

        mse = torch.mean(torch.cat(residuals))
        print("Overall MSE (voxel-wise):", mse.item())

    else:
        # WIP
        pass
    