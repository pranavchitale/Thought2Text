"""
Filename: evalEMwithGPT2.py
Author(s): 
- Rajath Rao (rajath.rao@stonybrook.edu)
- Pranav Chitale (pranavshailesh.chitale@stonybrook.edu)
- Ashutosh Tiwari (ashutosh.tiwari@stonybrook.edu)

Usage:
python3 decoding/evalEMwithGPT2.py --subject S1 --experiment perceived_speech --task wheretheressmoke

System Requirements:
- Operating System: Ubuntu
- Python Version: Python 3.10.14
- Dependencies: (conda) environment.yaml

Description:
This file trains the Encoder Model using fine-tuned GPT2.
The encoder serves to predict fMRI (BOLD) responses from textual stimulus for a given user. The learned parameters of the encoder allow
the autoregressive decoder LM (III. LM | Transformers) to be conditioned on the user's brain states (IV. Human Language Modeling).
The encoder is trained using bootstrapped regression (Line 78) (I. Syntax  |  Classification)
Saves resultant file to models/[subject]/encoding_model_[task].npz
"""


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
np.random.seed(42)


import torch, tqdm
import matplotlib.pyplot as plt
import transformers


parser = argparse.ArgumentParser()
parser.add_argument("--subject", type = str, required = True)
parser.add_argument("--gpt", type = str, default = "perceived")
parser.add_argument("--sessions", nargs = "+", type = int, default = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
args = parser.parse_args("--subject S1".split())


# training stories (inputs)
stories = []
with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
    sess_to_story = json.load(f) 
for sess in args.sessions:
    stories.extend(sess_to_story[str(sess)])


# converting vocab to ordered list and dict format to fit into author code
gpt2_tokenizer = transformers.GPT2Tokenizer.from_pretrained(os.path.join(config.DATA_LM_DIR, 'gpt2_tokenizer'))
gpt2_word_list = [None] * len(gpt2_tokenizer)
for token, idx, in gpt2_tokenizer.get_vocab().items():
    gpt2_word_list[idx] = token

gpt = GPT(path = os.path.join(config.DATA_LM_DIR, "gpt2_finetuned"), vocab = gpt2_word_list, word2id = gpt2_tokenizer.get_vocab(), device = config.GPT_DEVICE)

# using layer = 4, i.e. second last layer of distilgpt2 for extracting embeddings
features = LMFeatures(model = gpt, layer = 4, context_words = config.GPT_WORDS)

# estimate encoding model - rstim: X features (word embeddings) and rresp: y label (target embedding)
rstim, tr_stats, word_stats = get_stim(stories, features)
rresp = get_resp(args.subject, stories, stack = True)

# chunks for validation set
nchunks = int(np.ceil(rresp.shape[0] / 5 / config.CHUNKLEN))

# training linear regression model using 50 bootstraps
weights, alphas, bscorrs = bootstrap_ridge(rstim, rresp, use_corr = False, alphas = config.ALPHAS, nboots = config.NBOOTS, chunklen = config.CHUNKLEN, nchunks = nchunks)


bscorrs = bscorrs.mean(2).max(0)
vox = np.sort(np.argsort(bscorrs)[-config.VOXELS:])
stim_dict = {story : get_stim([story], features, tr_stats = tr_stats) for story in stories}
resp_dict = get_resp(args.subject, stories, stack = False, vox = vox)
noise_model = np.zeros([len(vox), len(vox)])


# noise model to simulate brain response
for hstory in stories:
    tstim, hstim = np.vstack([stim_dict[tstory] for tstory in stories if tstory != hstory]), stim_dict[hstory]
    tresp, hresp = np.vstack([resp_dict[tstory] for tstory in stories if tstory != hstory]), resp_dict[hstory]
    bs_weights = ridge(tstim, tresp, alphas[vox])
    resids = hresp - hstim.dot(bs_weights)
    bs_noise_model = resids.T.dot(resids)
    noise_model += bs_noise_model / np.diag(bs_noise_model).mean() / len(stories)


# save
save_location = os.path.join(config.MODEL_DIR, args.subject)
os.makedirs(save_location, exist_ok = True)
np.savez(os.path.join(save_location, "gpt2_encoding_model_%s" % args.gpt), 
    weights = weights, noise_model = noise_model, alphas = alphas, voxels = vox, stories = stories,
    tr_stats = np.array(tr_stats), word_stats = np.array(word_stats))