import torch
import spacy
from torchtext.datasets import Multi30k

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


# use iter-style datapipes

# 0. load datapipe from torchtext.dataset
train_iter = Multi30k(split='train')

# 1. build tokenizer
de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

# 2. build numerical vocab transform
def de_yield_tokens(data_iter):
    for de_sentence, en_sentence in data_iter:
        yield de_tokenizer(de_sentence)

def en_yield_tokens(data_iter):
    for de_sentence, en_sentence in data_iter:
        yield en_tokenizer(en_sentence)

unk_idx = 0
pad_idx = 1
bos_idx = 2
eos_idx = 3

special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

de_vocab_transform = build_vocab_from_iterator(de_yield_tokens(train_iter),
                                               min_freq=1,
                                               specials=special_symbols,
                                               special_first=True)

en_vocab_transform = build_vocab_from_iterator(en_yield_tokens(train_iter),
                                               min_freq=1,
                                               specials=special_symbols,
                                               special_first=True)
de_vocab_transform.set_default_index(unk_idx)
en_vocab_transform.set_default_index(unk_idx)

# 3. build tensor transform
def tensor_transform(token_ids):
    return torch.cat((torch.tensor([bos_idx]),
                      torch.tensor(token_ids),
                      torch.tensor([eos_idx])))


# Finally put together to form the collate function

def de_transform(sentence):
    # 1. token transform
    tokenized = de_tokenizer(sentence)
    # 2. numericalization 
    numericalized = de_vocab_transform(tokenized)
    # 3. add BOS/EOS and create tensor
    tensorized = tensor_transform(numericalized)
    return tensorized

def en_transform(sentence):
    # 1. token transform
    tokenized = en_tokenizer(sentence)
    # 2. numericalization 
    numericalized = en_vocab_transform(tokenized) 
    # 3. add BOS/EOS and create tensor
    tensorized = tensor_transform(numericalized)
    return tensorized

def collate_fn(batch):
    de_batch, en_batch = [], []
    for de_sample, en_sample in batch:
        de_batch.append(de_transform(de_sample.rstrip("\n")))
        en_batch.append(en_transform(en_sample.rstrip("\n")))

    de_batch = pad_sequence(de_batch, padding_value=pad_idx)
    en_batch = pad_sequence(en_batch, padding_value=pad_idx)
    return de_batch, en_batch





















