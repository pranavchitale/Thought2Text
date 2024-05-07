"""
Filename: train_distilgpt2_cnnDaily.py
Author(s): 
- Rajath Rao (rajath.rao@stonybrook.edu)                                                    
- Pranav Chitale (pranavshailesh.chitale@stonybrook.edu),                                   
- Ashutosh Tiwari (ashutosh.tiwari@stonybrook.edu) 


Usage:
(From root directory:)
$ python3 train-gpt2/train_distilgpt2_cnnDaily.py [-h] [-b [BATCHSZ]] [-l [LRNRATE]] [-w [WTDECAY]] [-e [EPOCHS]] [-s [SAVEPATH]]

System Requirements:
- Operating System: Ubuntu
- Python Version: Python 3.10.14
- Dependencies: (conda) environment.yaml

Description:
This python script is used to finetune a distilGPT2 model on the "cnn_dailymail" dataset  
consisting of more than 288,000 articles, with a mean token count of 786 tokens per       
article. After training, the model is stored at a either a "PATH" specified by the user   
or in this script's directory under the name "fineTunedDistilGPT2_cnnDaily_state_dict.pth"
The model is fine tuned following the ideas of Assignment 2, as seen in the "train_model" 
method.
This script takes in optional training hyperparameters from the command line, and a path
to store the trained model.
More details on the hyperparameters are in the README file.

"""

import re, argparse
from datasets import load_dataset

import numpy as np
import re

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from tqdm.auto import tqdm


import matplotlib.pyplot as plt

import transformers
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, get_scheduler

# https://huggingface.co/datasets/cnn_dailymail - Can/need to cite this
dataset = load_dataset('cnn_dailymail', '3.0.0')

modelname = 'distilbert/distilgpt2'
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('distilgpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('distilbert/distilgpt2')


# tokenizer = AutoTokenizer.from_pretrained(modelname)
# model = AutoModelForSeq2SeqLM.from_pretrained(modelname)
model = gpt2_model
tokenizer = gpt2_tokenizer
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Class to store the data
class cnnDataSet(Dataset):
    def __init__(self, data):
        self.data = data
        self.id = self.data['id']
        self.article = self.data['article']
        self.highlights = self.data['highlights']
        self.input_ids = torch.tensor(self.data['input_ids'])
        self.attention_mask = torch.tensor(self.data['attention_mask'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx],  self.attention_mask[idx], torch.sum(self.attention_mask[idx])

# Class to store the hyperparameters passed as cmd line args. 
class HyperParameters():
    def __init__(self, batch_sz=3, learn_rate = 1e-5, wt_decay = 1e-5, num_epochs = 2):
        self.batch_size = batch_sz
        self.num_epochs = num_epochs
        self.learn_rate = learn_rate
        self.wt_decay = wt_decay
        pass

    def get_hyperparam(self):
        """
        Method to return the hyper-parameters of training the model in this order:\\
        batch_size, learn_rate, weight_decay, num_epochs
        """
        return self.batch_size, self.learn_rate, self.wt_decay, self.num_epochs


# Most of the articles are of the format "METADATA (LOCATION) -- Article". Some
# (approx 10%) are not.
# This function ensures that the article text is split on the first '--', and then
# if the first string of the split matches the metadata format then the second
# split string is returned. Else, the entire article string is returned as is. 
def split_article(art):
    splitstr = re.split(' -- ', art, 1)
    # print(splitstr)
    if len(splitstr) == 1 or (re.search(r'(\([\w+]+\))|([A-Z][A-Z]+)|(\([\w\s]+\))|(\([\w]+\.[\w]+\))', splitstr[0])) is None:
        return art
    else:
        return splitstr[1]

    
def tokenize_str(examples):

    inputs = [split_article(art) for art in examples['article']]

    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, return_tensors='pt', padding='max_length')

    return model_inputs

# In this method, we use [III. Transformers] to finetune a DistilGPT2 model
def train_model(train_dataset, plot_loss_curves=True, model=gpt2_model, hyperparams=HyperParameters()):
    '''
    Method to fine tune and train the passed model
    Returns nothing
    '''

    # Hyperparameters
    batch_sz, learn_rate,  wt_decay, num_epochs, = hyperparams.get_hyperparam()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_sz)

    # torch.set_default_device(device)
    
    # model.to(dtype=torch.float16)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learn_rate, weight_decay=wt_decay)
    # lossfunc = binary_cross_entropy_with_logits().to(device)
    # print(model.parameters())
    dataloader_len = len(train_dataloader)
    num_training_steps = num_epochs * dataloader_len
    
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    progress_bar = tqdm(range(num_training_steps))
    loss_list = []

    # Use FP16 for speed purposes
    scaler = torch.cuda.amp.GradScaler()
    # Update loss and other values every 3 iterations
    accum_iter = 3
    
    torch.cuda.empty_cache()
    model.train()
    for epoch in range(num_epochs):
        
        epoch_loss = []
        for  batch_idx, batch in enumerate(train_dataloader):
            with torch.set_grad_enabled(True):


                max_valid_input, max_valid_attention = trunc_batch(batch)
                if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == dataloader_len):                    
                    optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    
                    output = model(
                                   input_ids = max_valid_input.to(device),
                                   attention_mask=max_valid_attention.to(device),
                                   labels=max_valid_input.to(device)
                                   )
                    loss = output.loss/accum_iter
                
                # loss.backward()
                scaler.scale(loss).backward()
                # model.float()
                if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == dataloader_len):
                    # optimizer.step()
                    scaler.step(optimizer)
                    # scaler.step(lr_scheduler)
                    # optimizer.zero_grad()
                    scaler.update()
                    lr_scheduler.step()
            
            progress_bar.update(1)
            if plot_loss_curves:
                loss_list.append(loss.item())
                
    if plot_loss_curves:
        plot_loss_curve(loss_list, f'distil-gpt2 LM LR : {learn_rate}')


def trunc_batch(batch):
    max_pad = torch.max(batch[2])
    # What we do here is that for all tensor in the batch, we truncate the attention and padding to the length of the 
    # maximum valid length, so save time and memory on computing the softmaxes
    input_ids = torch.stack([b[:max_pad] for b in batch[0]])
    attention_masks = torch.stack([b[:max_pad] for b in batch[1]])

    return input_ids, attention_masks

def plot_loss_curve(losslist, rword, save_title = 'loss_curve_DistilGPT2.png'):
    l = len(losslist)
    plt.plot(range(l), losslist)
    plt.title(f'Plotting the loss across Iterations for "{rword}" ')    
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    # plt.show()
    plt.savefig(save_title)


def main(hyperparams, model_paths):

    save_path = model_paths # Want to add load_path as well?
    
    # Just training the first 10k articles for now
    # 10k * 786 = approx 7M tokens to train on. 
    train = dataset['train'].select(range(10000))
    test = dataset['test'].select(range(1000))
    validation = dataset['validation']
    tokenized_train = train.map(tokenize_str, batched=True)
    tokenized_test = test.map(tokenize_str, batched=True)

    tokenized_train_dataset = cnnDataSet(tokenized_train)
    train_model(tokenized_train_dataset, plot_loss_curves=True, model=model, hyperparams=hyperparams)
    
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    print(f'Saving Model State as {save_path}')
    torch.save(model.state_dict(), save_path)
    

    
# Our parser for parsing cmd line arguments for hyperparameters. 
def initParser():
    parser = argparse.ArgumentParser(
        prog='train-gpt2/train_distilgpt2_cnnDaily.py',
        description='Python script to train distilbert/gpt2 on CNN-Daily-News dataset'
        )
    base_hpm = HyperParameters()
    batch_sz, learn_rate,  wt_decay, epochs = base_hpm.get_hyperparam()
    svpth = 'fineTunedDistilGPT2_cnnDaily_state_dict.pth'
    parser.add_argument('-b', '--batch-size', dest='batchsz', action='store', 
                        nargs='?',default=batch_sz, type=int, help=f'Batch Size. Default Value : {batch_sz}')
    parser.add_argument('-l', '--learn-rate', dest='lrnrate', action='store', 
                        nargs='?',default=learn_rate, type=float, help=f'Learning Rate. Default Value : {learn_rate}')
    parser.add_argument('-w', '--weight-decay', dest='wtdecay', action='store', 
                        nargs='?',default=wt_decay, type=float, help=f'Weight Decay. Default Value : {wt_decay}')
    parser.add_argument('-e', '--epochs', dest='epochs', action='store', 
                        nargs='?',default=epochs, type=int, help=f'Num Epochs. Default Value : {epochs}')
    parser.add_argument('-s', '--save-path', dest='savepath', action='store', 
                        nargs='?',default=svpth, type=str, help=f'Path to save model at. Default Path: {svpth}')
        
    return parser
if __name__ == '__main__':
    parser = initParser()
    args = parser.parse_args()
    args_hyperparams = HyperParameters(
        batch_sz=args.batchsz,
        learn_rate=args.lrnrate,
        wt_decay=args.wtdecay,
        num_epochs=args.epochs)
    
    # print(type(args.batchsz), type(args.epochs))
    # print(args_hyperparams.get_hyperparam())
    main(args_hyperparams, parser.savepath)
    