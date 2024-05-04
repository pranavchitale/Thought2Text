import os
import numpy as np
import json
import argparse
import torch

import config
from GPT import GPT
from StimulusModel import LMFeatures
from utils_stim import get_stim
from utils_resp import get_resp
from utils_ridge.ridge import ridge, bootstrap_ridge
np.random.seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, default = "S1")
    parser.add_argument("--gpt", type = str, default = "perceived", choices=["perceived", "imagined"])
    parser.add_argument("--sessions", nargs = "+", type = int, 
        default = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
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
    
    # load text stimulus
    rstim, tr_stats, word_stats = get_stim(stories, features)
    rstim = torch.from_numpy(rstim).float()

    # load fmri responses
    rresp = get_resp(args.subject, stories, stack = True)
    rresp = torch.from_numpy(rresp).float()

    # load EM weights
    load_location = os.path.join(config.MODEL_DIR, args.subject)
    encoding_model = np.load(os.path.join(load_location, "encoding_model_%s.npz" % args.gpt))
    weights = torch.from_numpy(encoding_model["weights"]).float()

    # make response predictions
    presp = torch.matmul(rstim, weights)

    # evaluate mse
    r2 = torch.square(presp - rresp)
    mse  = torch.mean(r2, dim=1)
    # print(mse.shape)

    overall_mse = torch.mean(mse)
    print("Overall MSE:", overall_mse.item())
    