# Thought2Text

We propose a framework aimed at assisting individuals with paralysis who are unable to speak. The motivation behind our research stems from the pressing need to provide these individuals with a means to express themselves effectively. We leverage functional magnetic resonance (fMRI) readings due to their non-invasive spatial and temporal information, as the input for a sequence-to-sequence task and decode them to corresponding text sequences, enabling autoregressive generation of semantic language from neural activity patterns.

In mid-2023, researchers at UT Austin conducted a study and devised a framework to predict the words that a subject was hearing or imagining. The system can interpret the recorded fMRI data and produce textual representations encapsulating the gist of what the subject heard [1]. An fMRI is a voxel-based data type that contains spatial and temporal information about brain activity, acquired through medical imaging techniques that measure changes in Blood Oxygen Level Dependent (BOLD) within the brain.

Humans speak at a rate of approximately 1-2 words per second, yet fMRI images possess a notably low temporal resolution (~0.5 Hz)[4]. We propose using a [IV. Applications] human language model with candidate sequence prediction. We aim to evaluate two baselines, one for the encoder and one for the decoder. The encoder predicts a subject's brain response to a text sequence, extracting [II. Semantics] temporal vector semantics and transforming them into another feature space. We then use [I. Syntax] regularized multivariate linear regression to predict the ground truth BOLD score for fMRI voxels. The decoder is an [III. Transformer LMs] autoregressive LM which is conditioned on both the textual context and BOLD ground truths. To improve our baseline models, we propose modifications including different semantic representations and non-linear regression heads on the encoder. All team members will play an equal role in building the baseline, while each team member will evaluate different variations of the proposed improvements.

### Our code:
Please consider the following files for grading:

`gpt2/train_distilgpt2_cnnDaily.py`

`semantic-decoding/decoding/EncodingModel.py`

`semantic-decoding/decoding/eval_encoders.py`

`semantic-decoding/decoding/evalEMwithGPT2.py`

`semantic-decoding/decoding/run_decoder.py`

`semantic-decoding/decoding/train_MLP.py`

`semantic-decoding/decoding/trainEMwithGPT2.py`

More details can be found in the comment headers in the above files.
