import torch
import random
import numpy as np
import itertools
from datetime import datetime
import json


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
def get_str_time():
    time = datetime.now()
    str_time = time.strftime('%Y-%m-%d-%H:%M:%S:%f')
    return str_time

def init_embedding_grad(full_embeddings):
    if full_embeddings.requires_grad:
        full_embeddings.grad.zero_()
    full_embeddings.requires_grad = True
    full_embeddings.retain_grad()
    
def iter_gen(l):
    # l [sent_num, update_num, evaluate_num]
    sent_num, update_num, evaluate_num = len(l), len(l[0]), len(l[0][0])
    new_seq = []
    for i in range(sent_num):
        list_tuple = tuple(l[i])
        all_permutions = list(itertools.product(*list_tuple))
        seqs = list(map(list, all_permutions))
        new_seq.append(seqs)

    return new_seq

def get_forbidden_tokens(tokenized_questions, tokenized_ans, tokenizer):
    ans_tok = []
    q_tok = []
    for item in tokenized_ans:
        ans_tok += item
    for item_list in tokenized_questions:
        for item in item_list:
            q_tok += item
    ans_tok = set(ans_tok)
    q_tok = set(q_tok)
    all_toks = set([i for i in range(tokenizer.vocab_size)])
    forbidden_toks = ans_tok - q_tok
    allowed_toks = all_toks - forbidden_toks
    allowed_toks = list(allowed_toks)
    forbidden_toks = list(forbidden_toks)
    forbidden_toks = torch.Tensor(forbidden_toks).long()
    allowed_toks = torch.Tensor(allowed_toks).long()
    return forbidden_toks, allowed_toks
