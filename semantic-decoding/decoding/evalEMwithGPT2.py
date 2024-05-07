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
This file takes the Encoder Model created in trainEMwithGPT2.py.
Additionally, it also requires the LM trained by GPT2_train.py.
"""




import os
import numpy as np
import json
import argparse
import h5py
from pathlib import Path

import config
from GPT import GPT
from Decoder import Decoder, Hypothesis
from LanguageModel import LanguageModel
from EncodingModel import EncodingModel
from StimulusModel import StimulusModel, get_lanczos_mat, affected_trs, LMFeatures
from utils_stim import predict_word_rate, predict_word_times

import transformers, tqdm



parser = argparse.ArgumentParser()
parser.add_argument("--subject", type = str, required = True)
parser.add_argument("--experiment", type = str, required = True)
parser.add_argument("--task", type = str, required = True)
# args = parser.parse_args("--subject S1 --experiment perceived_speech --task wheretheressmoke".split())
args = parser.parse_args()


# determine GPT checkpoint based on experiment
if args.experiment in ["imagined_speech"]: gpt_checkpoint = "imagined"
else: gpt_checkpoint = "perceived"

# determine word rate model voxels based on experiment
if args.experiment in ["imagined_speech", "perceived_movies"]: word_rate_voxels = "speech"
else: word_rate_voxels = "auditory"


hf = h5py.File(os.path.join(config.DATA_TEST_DIR, "test_response", args.subject, args.experiment, args.task + ".hf5"), "r")
resp = np.nan_to_num(hf["data"][:])
hf.close()


gpt2_tokenizer = transformers.GPT2Tokenizer.from_pretrained(os.path.join(config.DATA_LM_DIR, 'gpt2_tokenizer'))
gpt2_word_list = [None] * len(gpt2_tokenizer)
for token, idx, in gpt2_tokenizer.get_vocab().items():
    gpt2_word_list[idx] = token

gpt = GPT(path = os.path.join(config.DATA_LM_DIR, "gpt2_finetuned"), vocab = gpt2_word_list, word2id = gpt2_tokenizer.get_vocab(), device = config.GPT_DEVICE)
features = LMFeatures(model = gpt, layer = 4, context_words = config.GPT_WORDS)

with open(os.path.join(config.DATA_LM_DIR, "decoder_vocab.json"), "r") as f:
    decoder_vocab = json.load(f)

lm = LanguageModel(gpt, decoder_vocab, nuc_mass = config.LM_MASS, nuc_ratio = config.LM_RATIO)


# load models
load_location = os.path.join(config.MODEL_DIR, args.subject)
word_rate_model = np.load(os.path.join(load_location, "word_rate_model_%s.npz" % word_rate_voxels), allow_pickle = True)
encoding_model = np.load(os.path.join(load_location, "gpt2_encoding_model_%s.npz" % gpt_checkpoint))
weights = encoding_model["weights"]
noise_model = encoding_model["noise_model"]
tr_stats = encoding_model["tr_stats"]
word_stats = encoding_model["word_stats"]
em = EncodingModel(resp, weights, encoding_model["voxels"], noise_model, device = config.EM_DEVICE)
em.set_shrinkage(config.NM_ALPHA)
assert args.task not in encoding_model["stories"]


# predict word times
word_rate = predict_word_rate(resp, word_rate_model["weights"], word_rate_model["voxels"], word_rate_model["mean_rate"])
if args.experiment == "perceived_speech": word_times, tr_times = predict_word_times(word_rate, resp, starttime = -10)
else: word_times, tr_times = predict_word_times(word_rate, resp, starttime = 0)
lanczos_mat = get_lanczos_mat(word_times, tr_times)



# 169m 45.9s
# word_rate is the number of words spoken in an observation. List sums to total number of words in the transcript.
# word_times is the timestamp in seconds at which a word was spoken. Size = sum(word_rate).

# decode responses
decoder = Decoder(word_times, config.WIDTH) # WIDTH is beam size
sm = StimulusModel(lanczos_mat, tr_stats, word_stats[0], device = config.SM_DEVICE)
for sample_index in tqdm.tqdm(range(len(word_times))):
    trs = affected_trs(decoder.first_difference(), sample_index, lanczos_mat)
    ncontext = decoder.time_window(sample_index, config.LM_TIME, floor = 5)
    beam_nucs = lm.beam_propose(decoder.beam, ncontext)
    for c, (hyp, nextensions) in enumerate(decoder.get_hypotheses()):
        nuc, logprobs = beam_nucs[c]
        if len(nuc) < 1: continue
        extend_words = [hyp.words + [x] for x in nuc]
        extend_embs = list(features.extend(extend_words))
        stim = sm.make_variants(sample_index, hyp.embs, extend_embs, trs)
        likelihoods = em.prs(stim, trs)
        local_extensions = [Hypothesis(parent = hyp, extension = x) for x in zip(nuc, logprobs, extend_embs)]
        decoder.add_extensions(local_extensions, likelihoods, nextensions)

    decoder.extend(verbose = False)


if args.experiment in ["perceived_movie", "perceived_multispeaker"]: decoder.word_times += 10
save_location = os.path.join(config.RESULT_DIR, args.subject, args.experiment)
os.makedirs(save_location, exist_ok = True)
decoder.save(os.path.join(save_location, args.task+'_gpt2'))