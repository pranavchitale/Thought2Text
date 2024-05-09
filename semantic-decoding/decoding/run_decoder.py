"""
Filename: run_decoder.py
Author(s): 
- Rajath Rao (rajath.rao@stonybrook.edu)
- Pranav Chitale (pranavshailesh.chitale@stonybrook.edu)
- Ashutosh Tiwari (ashutosh.tiwari@stonybrook.edu)

Usage:
$ python semantic-decoding/decoding/run_decoder.py --stimulus BASE --variant BASE
$ python semantic-decoding/decoding/run_decoder.py --stimulus BASE --variant MLP --mlp_path semantic-decoding/models/S1/MLP_BASE_1e-3_1e-5.pth
$ python semantic-decoding/decoding/run_decoder.py --stimulus GPT2 --variant BASE
$ python semantic-decoding/decoding/run_decoder.py --stimulus GPT2 --variant MLP --mlp_path semantic-decoding/models/S1/MLP_GPT2_1e-3_1e-5.pth

System Requirements:
- Operating System: Ubuntu
- Python Version: Python 3.10.*
- Dependencies: (conda) environment.yaml

Description:
This file runs the Decoder. The options are `base` which uses baseline EncoderModel, `mlp` which uses a variant EncoderModel with our
trained MLP, and `gpt2` which uses the fine-tuned `distilgpt2` (III. LM | Transformers) to extract stimulus features which are 
semantic vectors (II. Semantics | Probabilistic Models) on Line 74. The decoder uses the pretrained weights from the encoder variants
to autoregressively propose candidate sequence tokens (III. LM | Transformers) while being conditioned on the user's brain state
which can be seen starting from Line 100. This is also an Additionally, the language generation technique uses a Beam Search (III. LM | Transformers)
with nucleus sampling--selecting from a subset of the most likely next tokens in a language model based on their probabilities rather
than sampling from the entire vocabulary (Line 105+).
"""


import os
import numpy as np
import json
import argparse
import h5py
from pathlib import Path
import transformers

import config
from GPT import GPT
from Decoder import Decoder, Hypothesis
from LanguageModel import LanguageModel
from EncodingModel import EncodingModel
from StimulusModel import StimulusModel, get_lanczos_mat, affected_trs, LMFeatures
from utils_stim import predict_word_rate, predict_word_times


# Experiments:
# CUDA_VISIBLE_DEVICES=3 python semantic-decoding/decoding/run_decoder.py --stimulus BASE --variant BASE
#   Done.
# CUDA_VISIBLE_DEVICES=3 python semantic-decoding/decoding/run_decoder.py --stimulus BASE --variant MLP --mlp_path semantic-decoding/models/S1/MLP_BASE_1e-3_1e-5.pth
#   Done.
# CUDA_VISIBLE_DEVICES=3 python semantic-decoding/decoding/run_decoder.py --stimulus GPT2 --variant BASE
#   To Do.
# CUDA_VISIBLE_DEVICES=3 python semantic-decoding/decoding/run_decoder.py --stimulus GPT2 --variant MLP --mlp_path semantic-decoding/models/S1/MLP_GPT2_1e-3_1e-5.pth
#   To Do.
# CUDA_VISIBLE_DEVICES=3 python semantic-decoding/decoding/run_decoder.py --experiment imagined_speech --task alpha_repeat-1 --stimulus BASE --variant BASE
#   To Do.
# CUDA_VISIBLE_DEVICES=3 python semantic-decoding/decoding/run_decoder.py --experiment imagined_speech --task alpha_repeat-1 --stimulus BASE --variant MLP --mlp_path semantic-decoding/models/S1/MLP_BASE_1e-3_1e-5.pth
#   To Do.
# CUDA_VISIBLE_DEVICES=3 python semantic-decoding/decoding/run_decoder.py --experiment imagined_speech --task alpha_repeat-1 --stimulus GPT2 --variant BASE
#   To Do.
# CUDA_VISIBLE_DEVICES=3 python semantic-decoding/decoding/run_decoder.py --experiment imagined_speech --task alpha_repeat-1 --stimulus GPT2 --variant MLP --mlp_path semantic-decoding/models/S1/MLP_GPT2_1e-3_1e-5.pth
#   To Do.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, default = "S1")
    parser.add_argument("--experiment", type = str, default = "perceived_speech", choices=["perceived_speech", "imagined_speech"])
    parser.add_argument("--task", type = str, default = "wheretheressmoke")
    parser.add_argument("--variant", type = str, default = "BASE", choices = ["BASE", "MLP"])
    parser.add_argument("--stimulus", type = str, default = "BASE", choices = ["BASE", "GPT2"])
    parser.add_argument("--mlp_path", type = str, default = "", help = "Specify path to checkpoint file if `mlp` variant is selected")
    args = parser.parse_args()
    
    # determine GPT checkpoint based on experiment
    if args.experiment in ["imagined_speech"]: gpt_checkpoint = "imagined"
    else: gpt_checkpoint = "perceived"

    # determine word rate model voxels based on experiment
    if args.experiment in ["imagined_speech", "perceived_movies"]: word_rate_voxels = "speech"
    else: word_rate_voxels = "auditory"

    # load responses
    hf = h5py.File(os.path.join(config.DATA_TEST_DIR, "test_response", args.subject, args.experiment, args.task + ".hf5"), "r")
    resp = np.nan_to_num(hf["data"][:])
    hf.close()
    
    # load gpt
    if args.stimulus == 'GPT2':
        gpt2_tokenizer = transformers.GPT2Tokenizer.from_pretrained('distilgpt2', cache_dir='cache/')
        gpt2_word_list = [None] * len(gpt2_tokenizer)
        for token, idx, in gpt2_tokenizer.get_vocab().items():
            gpt2_word_list[idx] = token
        gpt2_pretrained_path = 'gpt2/models/gpt2_200_0.0005_1e-05' # change this path if needed
        gpt = GPT(path = gpt2_pretrained_path, vocab = gpt2_word_list, word2id = gpt2_tokenizer.get_vocab(), device = config.GPT_DEVICE)
        features = LMFeatures(model = gpt, layer = 4, context_words = config.GPT_WORDS)
        with open(os.path.join(config.DATA_LM_DIR, "decoder_vocab.json"), "r") as f:
            decoder_vocab = json.load(f)
        lm = LanguageModel(gpt, decoder_vocab, nuc_mass = config.LM_MASS, nuc_ratio = config.LM_RATIO)
    else:
        with open(os.path.join(config.DATA_LM_DIR, gpt_checkpoint, "vocab.json"), "r") as f:
            gpt_vocab = json.load(f)
        with open(os.path.join(config.DATA_LM_DIR, "decoder_vocab.json"), "r") as f:
            decoder_vocab = json.load(f)
        gpt = GPT(path = os.path.join(config.DATA_LM_DIR, gpt_checkpoint, "model"), vocab = gpt_vocab, device = config.GPT_DEVICE)
        features = LMFeatures(model = gpt, layer = config.GPT_LAYER, context_words = config.GPT_WORDS)
        lm = LanguageModel(gpt, decoder_vocab, nuc_mass = config.LM_MASS, nuc_ratio = config.LM_RATIO)

    # load models
    load_location = os.path.join(config.MODEL_DIR, args.subject)
    word_rate_model = np.load(os.path.join(load_location, "word_rate_model_%s.npz" % word_rate_voxels), allow_pickle = True)
    encoding_model = np.load(os.path.join(load_location, f"encoder_{gpt_checkpoint}_{args.stimulus}.npz"))
    weights = encoding_model["weights"]
    noise_model = encoding_model["noise_model"]
    tr_stats = encoding_model["tr_stats"]
    word_stats = encoding_model["word_stats"]
    if args.variant == "BASE":
        em = EncodingModel(resp, weights, encoding_model["voxels"], noise_model, device = config.EM_DEVICE)
    elif args.variant == "MLP":
        em = EncodingModel(resp, weights, encoding_model["voxels"], noise_model, mlp_path = args.mlp_path, device = config.EM_DEVICE)
    em.set_shrinkage(config.NM_ALPHA)
    assert args.task not in encoding_model["stories"]
    
    # predict word times
    word_rate = predict_word_rate(resp, word_rate_model["weights"], word_rate_model["voxels"], word_rate_model["mean_rate"])
    if args.experiment == "perceived_speech": word_times, tr_times = predict_word_times(word_rate, resp, starttime = -10)
    else: word_times, tr_times = predict_word_times(word_rate, resp, starttime = 0)
    lanczos_mat = get_lanczos_mat(word_times, tr_times)

    # decode responses
    decoder = Decoder(word_times, config.WIDTH)
    sm = StimulusModel(lanczos_mat, tr_stats, word_stats[0], device = config.SM_DEVICE)
    for sample_index in range(len(word_times)):
        # print(f'sample_index: {sample_index}')
        first_diff = decoder.first_difference()
        # print(f'diff: {first_diff}')
        trs = affected_trs(first_diff, sample_index, lanczos_mat)
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
    decoder.save(os.path.join(save_location, f"{args.task}_{args.stimulus}_{args.variant}"))