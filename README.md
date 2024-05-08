# Thought2Text

Base code: https://github.com/HuthLab/semantic-decoding (cite)

We propose a framework aimed at assisting individuals with paralysis who are unable to speak. The motivation behind our research stems from the pressing need to provide these individuals with a means to express themselves effectively. We leverage functional magnetic resonance (fMRI) readings due to their non-invasive spatial and temporal information, as the input for a sequence-to-sequence task and decode them to corresponding text sequences, enabling autoregressive generation of semantic language from neural activity patterns.

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

*_Python notebooks were used purely for testing purposes, please do not consider them as official code for grading. Experiments can be found there however._
