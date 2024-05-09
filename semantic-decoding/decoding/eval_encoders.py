"""
Filename: eval_encoders.py
Author(s): 
- Rajath Rao (rajath.rao@stonybrook.edu)
- Pranav Chitale (pranavshailesh.chitale@stonybrook.edu)
- Ashutosh Tiwari (ashutosh.tiwari@stonybrook.edu)

Usage:
$ python semantic-decoding/decoding/eval_encoders.py --stimulus BASE --variant BASE
$ python semantic-decoding/decoding/eval_encoders.py --stimulus BASE --variant MLP --mlp_path semantic-decoding/models/S1/MLP_BASE_1e-3_1e-5.pth
$ python semantic-decoding/decoding/eval_encoders.py --stimulus GPT2 --variant BASE
$ python semantic-decoding/decoding/eval_encoders.py --stimulus GPT2 --variant MLP --mlp_path semantic-decoding/models/S1/MLP_GPT2_1e-3_1e-5.pth

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
import transformers

import config
from train_MLP import FMRIDataset, MLP, calc_loss
from GPT import GPT
from StimulusModel import LMFeatures
from utils_stim import get_stim
from utils_resp import get_resp


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
np.random.seed(42)

# CUDA_VISIBLE_DEVICES=1 python semantic-decoding/decoding/eval_encoders.py --stimulus BASE --variant BASE
#   Overall MSE (voxel-wise): 0.9393559098243713
#   Overall MAE (voxel-wise): 0.7574178576469421
# CUDA_VISIBLE_DEVICES=1 python semantic-decoding/decoding/eval_encoders.py --stimulus BASE --variant MLP --mlp_path semantic-decoding/models/S1/MLP_BASE_1e-3_1e-5.pth
#   Overall MSE (voxel-wise): 0.7740427255630493
#   Overall MAE (voxel-wise): 0.6959194540977478
# CUDA_VISIBLE_DEVICES=1 python semantic-decoding/decoding/eval_encoders.py --stimulus GPT2 --variant BASE
#   Overall MSE (voxel-wise): 0.978938639163971
#   Overall MAE (voxel-wise): 0.7733919620513916
# CUDA_VISIBLE_DEVICES=1 python semantic-decoding/decoding/eval_encoders.py --stimulus GPT2 --variant MLP --mlp_path semantic-decoding/models/S1/MLP_GPT2_100_0.0005_1e-05.pth
#   Overall MSE (voxel-wise): 0.692783534526825
#   Overall MAE (voxel-wise): 0.6567291617393494

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, default = "S1")
    parser.add_argument("--gpt", type = str, default = "perceived", choices=["perceived", "imagined"])
    parser.add_argument("--sessions", nargs = "+", type = int, 
        default = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    parser.add_argument("--stimulus", type = str, default = "BASE", choices = ["BASE", "GPT2"])
    parser.add_argument("--variant", type = str, default = "BASE", choices=["BASE", "MLP"])
    parser.add_argument("--mlp_path", type = str, help="Specify path to load MLP checkpoint...")
    args = parser.parse_args()

    # training stories
    stories = []
    with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f)
    for sess in args.sessions:
        stories.extend(sess_to_story[str(sess)])

    # load gpt
    if args.stimulus == 'BASE':
        with open(os.path.join(config.DATA_LM_DIR, args.gpt, "vocab.json"), "r") as f:
            gpt_vocab = json.load(f)
        gpt = GPT(path = os.path.join(config.DATA_LM_DIR, args.gpt, "model"), vocab = gpt_vocab, device = config.GPT_DEVICE)
        features = LMFeatures(model = gpt, layer = config.GPT_LAYER, context_words = config.GPT_WORDS)
    else:
        gpt2_tokenizer = transformers.GPT2Tokenizer.from_pretrained('distilgpt2', cache_dir='cache/')
        gpt2_word_list = [None] * len(gpt2_tokenizer)
        for token, idx, in gpt2_tokenizer.get_vocab().items():
            gpt2_word_list[idx] = token
        gpt2_pretrained_path = 'gpt2/models/gpt2_200_0.0005_1e-05' # change this path if needed
        gpt = GPT(path = gpt2_pretrained_path, vocab = gpt2_word_list, word2id = gpt2_tokenizer.get_vocab(), device = config.GPT_DEVICE)
        features = LMFeatures(model = gpt, layer = 4, context_words = config.GPT_WORDS)
    
    # Prepare stimulus + response data
    em_cp = np.load(os.path.join(config.MODEL_DIR, args.subject, f"encoder_{args.gpt}_{args.stimulus}.npz"))
    rstim, tr_stats, word_stats = get_stim(stories, features)
    rresp = get_resp(args.subject, stories, stack = True)[:, em_cp["voxels"]]
    train_dataset = FMRIDataset(rstim, rresp)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)

    print(f"Beginning Evaluation with {args.variant}")
    if args.variant == "BASE":
        rstim = torch.from_numpy(rstim).float().to(DEVICE)
        rresp = torch.from_numpy(rresp).float().to(DEVICE)

        # Load weights
        weights = torch.from_numpy(em_cp["weights"])[:, em_cp["voxels"]].float().to(DEVICE)

        # get response predictions
        presp = torch.matmul(rstim, weights)

        # calculate error
        residual = presp - rresp

        mse = torch.square(residual)
        mse  = torch.mean(mse, dim=1)
        mse = torch.mean(mse)
        print("Overall MSE (voxel-wise):", mse.item())

        mae = torch.abs(residual)
        mae  = torch.mean(mae, dim=1)
        mae = torch.mean(mae)
        print("Overall MAE (voxel-wise):", mae.item())

    elif args.variant == "MLP":
        model = MLP(3072, config.VOXELS).to(DEVICE)

        # Load weights
        model_state_dict = torch.load(args.mlp_path, map_location=DEVICE)
        if "module." in list(model_state_dict.keys())[0]:
            model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
        model.load_state_dict(model_state_dict)

        residuals_2 = []
        residuals_a = []
        model.eval()
        for idx, batch in enumerate(train_loader):
            rstim, rresp = batch[0].to(DEVICE), batch[1].to(DEVICE)
            presp = model(rstim)
            
            # calculate error
            res = presp - rresp
            r2 = torch.square(res)
            ra = torch.abs(res)
            residuals_2.append(torch.mean(r2, dim=1))
            residuals_a.append(torch.mean(ra, dim=1))

        residuals_2 = torch.cat(residuals_2)
        mse = torch.mean(residuals_2)
        print("Overall MSE (voxel-wise):", mse.item())
        residuals_a = torch.cat(residuals_a)
        mae = torch.mean(residuals_a)
        print("Overall MAE (voxel-wise):", mae.item())

    