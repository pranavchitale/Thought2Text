# Code to train GPT2 Model on CNN Dataset

Dataset used: https://huggingface.co/datasets/cnn_dailymail - (Cite)

```
>$ python3 train_distilgpt2_cnnDaily.py -h
usage: gpt2/train_distilgpt2_cnnDaily.py [-h] [-b [BATCHSZ]] [-l [LRNRATE]] [-w [WTDECAY]] [-e [EPOCHS]] [-s [SAVEPATH]]

Python script to train distilbert/gpt2 on CNN-Daily-News dataset

options:
  -h, --help            show this help message and exit
  -b [BATCHSZ], --batch-size [BATCHSZ]
                        Batch Size. Default Value : 3
  -l [LRNRATE], --learn-rate [LRNRATE]
                        Learning Rate. Default Value : 1e-05
  -w [WTDECAY], --weight-decay [WTDECAY]
                        Weight Decay. Default Value : 1e-05
  -e [EPOCHS], --epochs [EPOCHS]
                        Num Epochs. Default Value : 2
  -s [SAVEPATH], --save-path [SAVEPATH]
                        Path to save model at. Default Path: fineTunedDistilGPT2_cnnDaily_state_dict.pth
```