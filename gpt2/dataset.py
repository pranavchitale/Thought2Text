import re, math
import torch
from transformers.tokenization_utils_base import BatchEncoding


# Dataset class for HuggingFace CNN articles data
class CNNDataset(torch.utils.data.Dataset):
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
        return self.input_ids[idx], self.attention_mask[idx], torch.sum(self.attention_mask[idx])


class StoryDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, tokenizer):
        self.docs = []
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        
        pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        for line in lines:
            tokens = tokenizer(line, max_length=1024, padding='max_length', return_tensors='pt')
            num_tokens = len(tokens['input_ids'][0])
            if num_tokens > 1024:
                num_batches = math.ceil(num_tokens / 1024)
                for i in range(num_batches):
                    input_ids_batch = torch.full((1, 1024), pad_token_id, dtype=torch.int64)
                    attention_mask_batch = torch.zeros((1, 1024), dtype=torch.int64)
                    input_ids = tokens['input_ids'][0][1024*i:1024*(i+1)]
                    attention_mask = tokens['attention_mask'][0][1024*i:1024*(i+1)]
                    for j in range(len(input_ids)):
                        input_ids_batch[0][j] = input_ids[j]
                        attention_mask_batch[0][j] = attention_mask[j]
                    batch_encoding = BatchEncoding({"input_ids": input_ids_batch, "attention_mask": attention_mask_batch})
                    self.docs.append(batch_encoding)
            else:
                self.docs.append(tokens)

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        return self.docs[idx]