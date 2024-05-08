import os
import argparse
import numpy as np
import torch
from tqdm import tqdm 
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, get_scheduler
from matplotlib import pyplot as plt
from dataset import StoryDataset

# CUDA_VISIBLE_DEVICES=0,1,2 python gpt2/train.py --num_epochs 150 --batch_size 8 --lr 5e-4 --wd 1e-5 > gpt2/exps/gpt2_150_5e-4_1e-5.txt

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train(model, train, num_epochs, batch_size, lr, wd, plot=True):
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, num_workers=24, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    print('Starting Training Loop...')
    model.train()
    losses = []
    for epoch in range(num_epochs):
        batch_losses = []
        batches = tqdm(train_loader)
        for idx, batch in enumerate(batches):
            input_ids, attention_mask = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE)
            with torch.set_grad_enabled(True):
                with torch.cuda.amp.autocast():
                    output = model(input_ids = input_ids, attention_mask = attention_mask, labels = input_ids)
                    loss = output.loss.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            del input_ids, attention_mask
            torch.cuda.empty_cache()
            print(f'\tBatch: {idx + 1}/{len(batches)}\tBatch Loss: {batch_losses[-1]}')
        losses.extend(batch_losses)
        print(f'Epoch {epoch + 1}/{args.num_epochs}\tAverage Loss: {np.mean(batch_losses)}\t')
    
    if plot:
        plot_loss_curve(losses, num_epochs, lr, wd)


def plot_loss_curve(losses, e, lr, wd):
    plt.plot(range(len(losses)), losses, color='royalblue')
    plt.title(f'Training Loss Curve (lr: {lr:.5f}, wd: {wd:.5})')
    plt.xlabel('Iteration Steps')
    plt.ylabel('Loss')
    plt.savefig(f'gpt2/visuals/gpt2_loss_{e}_{lr}_{wd}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num_epochs', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=2)
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-5)
    parser.add_argument('--wd', '--weight_decay', type=float, default=1e-5)
    parser.add_argument('--load_path', type=str)
    args = parser.parse_args()

    # Load GPT2 Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('distilgpt2', cache_dir='cache/')
    tokenizer.pad_token = tokenizer.eos_token # token_id: 50256
    
    # Prepare datasets
    dataset = StoryDataset('semantic-decoding/data_lm/stories.txt', tokenizer)
    total_length = len(dataset)
    train_length = int(0.8 * total_length)
    val_length = total_length - train_length
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_length, val_length])

    # Load the pre-trained model
    model = GPT2LMHeadModel.from_pretrained('distilbert/distilgpt2', cache_dir="cache/").to(DEVICE)
    if args.load_path:
        model_state_dict = torch.load(args.load_path, map_location=DEVICE)
        if "module." in list(model_state_dict.keys())[0]:
            model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
        model.load_state_dict(model_state_dict)
        print('Resuming model training from checkpoint...')

    # DataParallel multi-GPU training (if available)
    if torch.cuda.is_available():
        devices = list(range(torch.cuda.device_count()))
        if len(devices) > 1:
            model = torch.nn.DataParallel(model, device_ids=devices)

    # Fine-tune `distilGPT2`
    train(model, train_dataset, args.num_epochs, args.batch_size, args.lr, args.wd)
    print('Done Training...\n')

    # Save the model state
    print('Saving the model states...')
    save_path = f'gpt2/models/gpt2_{args.num_epochs}_{args.lr}_{args.wd}.pth'
    model.cpu()
    torch.cuda.empty_cache()
    os.makedirs('gpt2/models/', exist_ok = True)
    torch.save(model.state_dict(), save_path)
    print(f'\tSaved to location: {save_path}')

