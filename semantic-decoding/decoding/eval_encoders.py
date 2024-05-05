import os
import numpy as np
import json
import argparse
import torch
from matplotlib import pyplot as plt

import config
from train_MLP import FMRIDataset, MLP, calc_loss
from GPT import GPT
from StimulusModel import LMFeatures
from utils_stim import get_stim
from utils_resp import get_resp
from utils_ridge.ridge import ridge, bootstrap_ridge


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
np.random.seed(42)

# EM_BASE (voxel-wise MSE): 1.3242875337600708
# EM_MLP (voxel-wise MSE): 0.9773797988891602

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, default = "S1")
    parser.add_argument("--gpt", type = str, default = "perceived", choices=["perceived", "imagined"])
    parser.add_argument("--sessions", nargs = "+", type = int, 
        default = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    parser.add_argument("--model", type = str, default = "EM_BASE", choices=["EM_BASE", "EM_MLP", "EM_GPT2"])
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
    rstim, tr_stats, word_stats = get_stim(stories, features)
    rresp = get_resp(args.subject, stories, stack = True)
    train_dataset = FMRIDataset(rstim, rresp)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)

    print(f"Beginning Evaluation with {args.model}")
    if args.model == "EM_BASE":
        rstim = torch.from_numpy(rstim).float().to(DEVICE)
        rresp = torch.from_numpy(rresp).float().to(DEVICE)

        # Load weights
        encoding_model = np.load(args.load_path)
        weights = torch.from_numpy(encoding_model["weights"]).float().to(DEVICE)

        # get response predictions
        presp = torch.matmul(rstim, weights) * 10

        # calculate error
        mse = calc_loss(presp, rresp)
        print("Overall MSE (voxel-wise):", mse.item())

    elif args.model == "EM_MLP":
        model = MLP(3072, 3072, 81126).to(DEVICE)

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
        pass
    