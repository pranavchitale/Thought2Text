# Code to train GPT2 Model on CNN Dataset

Dataset used: https://huggingface.co/datasets/cnn_dailymail - Can/need to cite this

```
>$ python3 train_distilgpt2_cnnDaily.py -h
usage: train-gpt2/train_distilgpt2_cnnDaily.py [-h] [-b [BATCHSZ]] [-l [LRNRATE]] [-w [WTDECAY]] [-e [EPOCHS]]

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
```