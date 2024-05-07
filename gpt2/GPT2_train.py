"""
Filename: evalEMwithGPT2.py
Author(s): 
- Rajath Rao (rajath.rao@stonybrook.edu)
- Pranav Chitale (pranavshailesh.chitale@stonybrook.edu)
- Ashutosh Tiwari (ashutosh.tiwari@stonybrook.edu)

Usage:
python3 gpt/GPT2_train.py -b 8 -l 5e-7 -w 1e-2 -e 2

System Requirements:
- Operating System: Ubuntu
- Python Version: Python 3.10.14
- Dependencies: (conda) environment.yaml

Description:
Fine-tunes DistilGPT2 on CNN DailyMail dataset.
"""


import re, argparse, sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, get_scheduler

import os, json, transformers

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

def split_article(art):
    splitstr = re.split(' -- ', art, 1)
    # print(splitstr)
    if len(splitstr) == 1 or (re.search(r'(\([\w+]+\))|([A-Z][A-Z]+)|(\([\w\s]+\))|(\([\w]+\.[\w]+\))', splitstr[0])) is None:
        return art
    else:
        return splitstr[1]

    
def tokenize_str(examples):
    # inputs = [doc for doc in examples["article"]]
    inputs = []
    # labs = examples['highlights']
    
    inputs = [split_article(art) for art in examples['article']]

    # model_inputs = gpt2_tokenizer(inputs, max_length=1024, truncation=True)
    # labels = gpt2_tokenizer(text_target=labs, max_length=128, truncation=True)

    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, return_tensors='pt', padding='max_length')
    # labels = tokenizer(text_target=labs, max_length=128, truncation=True)

    # model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train_model(train_dataset, plot_loss_curves=True, model=gpt2_model, hyperparams=HyperParameters()):
    '''
    Method to fine tune and train the passed model
    Returns nothing
    train_dataset - The datset whose text is in the format PASSAGE\nQUESTION?ANSWER
    '''

    # train_dataset = GPT2CustomDataset(training_text, training_labels, gpt2_tokenizer)
    # unique_labels = train_dataset.get_unique_labels()
    # unique_labels_encoding = {lab:gpt2_tokenizer.encode(lab)[0] for lab in unique_labels}
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

def plot_loss_curve(losslist, rword, save_title = 'loss_curve_a2_p2.png'):
    l = len(losslist)
    plt.plot(range(l), losslist)
    plt.title(f'Plotting the loss across Iterations for "{rword}" ')    
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    # plt.show()
    plt.savefig(save_title)


def main(hyperparams):
    # Just training the first 10k articles for now
    train = dataset['train'].select(range(10000))
    test = dataset['test'].select(range(1000))
    validation = dataset['validation']
    tokenized_train = train.map(tokenize_str, batched=True)
    tokenized_test = test.map(tokenize_str, batched=True)

    tokenized_train_dataset = cnnDataSet(tokenized_train)
    train_model(tokenized_train_dataset, plot_loss_curves=True, model=model, hyperparams=hyperparams)
    
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    torch.save(model, 'fineTunedDistilGPT2_cnnDaily')

def initParser():
    parser = argparse.ArgumentParser(
        prog='train-gpt2/train_distilgpt2_cnnDaily.py',
        description='Python script to train distilbert/gpt2 on CNN-Daily-News dataset'
        )
    base_hpm = HyperParameters()
    batch_sz, learn_rate,  wt_decay, epochs, = base_hpm.get_hyperparam()
    parser.add_argument('-b', '--batch-size', dest='batchsz', action='store', 
                        nargs='?',default=batch_sz, type=int, help=f'Batch Size. Default Value : {batch_sz}')
    parser.add_argument('-l', '--learn-rate', dest='lrnrate', action='store', 
                        nargs='?',default=learn_rate, type=float, help=f'Learning Rate. Default Value : {learn_rate}')
    parser.add_argument('-w', '--weight-decay', dest='wtdecay', action='store', 
                        nargs='?',default=wt_decay, type=float, help=f'Weight Decay. Default Value : {wt_decay}')
    parser.add_argument('-e', '--epochs', dest='epochs', action='store', 
                        nargs='?',default=epochs, type=int, help=f'Num Epochs. Default Value : {epochs}')
        
    return parser



if __name__ == '__main__':

    parser = initParser()
    args = parser.parse_args()
    args_hyperparams = HyperParameters(
        batch_sz=args.batchsz,
        learn_rate=args.lrnrate,
        wt_decay=args.wtdecay,
        num_epochs=args.epochs)


    # load data
    dataset = load_dataset('cnn_dailymail', '3.0.0')

    # load model
    modelname = 'distilbert/distilgpt2'
    tokenizer = transformers.GPT2Tokenizer.from_pretrained(modelname, eos_token=None, bos_token=None, pad_token='<pad>', unk_token='<unk>', add_prefix_space=False)
    model = GPT2LMHeadModel.from_pretrained(modelname)


    with open(os.path.join('/workspace/Thought2Text/semantic-decoding/data_lm/', 'perceived', "vocab.json"), "r") as f:
        gpt_vocab = json.load(f)

    # adding tokens from author's gpt1 to have compatibility with stimuli stories
    tokenizer.add_tokens(gpt_vocab)

    # update model dims to match new vocab size
    model.resize_token_embeddings(len(tokenizer))


    # Just training the first 20k articles for now
    train = dataset['train'].select(range(20000))
    test = dataset['test'].select(range(2000))
    validation = dataset['validation']
    tokenized_train = train.map(tokenize_str, batched=True)
    tokenized_test = test.map(tokenize_str, batched=True)

    tokenized_train_dataset = cnnDataSet(tokenized_train)

    # train model
    train_model(tokenized_train_dataset, plot_loss_curves=True, model=model, hyperparams=args_hyperparams)

    # save model and tokenizer for use in trainEMwithGPT2.py and evalEMwithGPT2.py
    model.save_pretrained('gpt2_finetuned')
    tokenizer.save_pretrained('gpt2_tokenizer')