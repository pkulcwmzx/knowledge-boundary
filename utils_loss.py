import torch
from torch import nn
from sentence_transformers.util import normalize_embeddings
from torch.nn import functional as F

def log_prob_loss(logits, labels, loss_fct, temp=1):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_logits = shift_logits / temp
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

def log_perplexity_loss(logits, prompts, question_len):
    shift_prompts = prompts[:, 1:]
    shift_logits = logits[:, :shift_prompts.shape[1], :]
    log_probs = F.log_softmax(shift_logits, dim=2)
    stacked_perplexities = torch.stack(
        [log_probs[i, torch.arange(question_len[i]-1), shift_prompts[i, :question_len[i]-1]].mean() for i in
         range(log_probs.shape[0])])

    return -stacked_perplexities.mean()

def similarity_loss(logits, ref_logits, question_len):
    probs = F.softmax(logits, dim=2)
    input_logits = torch.stack([probs[i, question_len[i], :] for i in range(probs.shape[0])])
    loss_fct = nn.CosineEmbeddingLoss(reduction='mean')
    tgt = torch.Tensor([1] * probs.shape[0]).long().to("cuda")
    similarity = loss_fct(input_logits, ref_logits, tgt)

    return similarity

def get_prob_loss(out, labels, loss_fn):
    logits = out.logits
    loss_fct = nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fn(logits, labels, loss_fct)
    return loss


def get_loss(out, labels, prompts, question_len):
    logits = out.logits
    loss_fct = nn.CrossEntropyLoss(reduction='mean')
    loss = log_prob_loss(logits, labels, loss_fct)

    return loss


def get_perp_loss(out, prompts, question_len):
    logits = out.logits
    perp_loss = log_perplexity_loss(logits, prompts, question_len)
    return perp_loss

def get_semantic_loss(semantic_feature, now_semantic_feature, device):
    # calculate the semantic loss between the semantic feature of the original question and the generated question
    # semantic_feature [bsz, seq_len, embed_dim]
    # now_semantic_feature [bsz, seq_len, embed_dim]
    loss_fct = nn.CosineEmbeddingLoss(reduction='mean')
    tgt = torch.Tensor([1] * semantic_feature.shape[0]).long().to(device)
    similarity = loss_fct(semantic_feature[:, -1, :], now_semantic_feature[:, -1, :], tgt)

    return similarity


def get_target_loss(logits, ans, device):
    losses = 0
    for i in range(logits.shape[0]):
        loss = float("inf")
        for tokenized_ans in ans:
            tokenized_ans = torch.Tensor(tokenized_ans).long().to(device)
            l = cal_target_loss(logits[i, :, :], tokenized_ans)
            loss = min(l, loss)

        losses += loss

    return losses / logits.shape[0]


def cal_target_loss(logits, tokenized_ans):
    # calculate the target loss between the generated question and the original answer
    # logits [generate_len, vocab_size]
    # tokenized_ans [ans_len]
    # window loss, return min loss
    loss_fct = nn.CrossEntropyLoss(reduction='mean')
    # loss = log_prob_loss(logits, tokenized_ans, loss_fct)
    ans_len = tokenized_ans.shape[0]

    if ans_len > logits.shape[0]:
        return float("inf")

    # tokenized_ans = tokenized_ans.unsqueeze(0).repeat(logits.shape[0], 1)
    for i in range(logits.shape[0] - ans_len):
        loss = loss_fct(logits[i:i + ans_len, :], tokenized_ans.reshape(-1))
        # loss = log_prob_loss(logits[:, i:i+ans_len, :], tokenized_ans, loss_fct)
        if i == 0:
            min_loss = loss
        else:
            if loss < min_loss:
                min_loss = loss
    return min_loss

def gradient_reg(input_embeds, embedding_table, allowed_toks):
    curr_embeds = input_embeds.reshape(-1, input_embeds.shape[-1])
    curr_embeds = normalize_embeddings(curr_embeds)
    embedding_table = normalize_embeddings(embedding_table)
    allowed_embeddings = embedding_table[allowed_toks, :]
    dist = torch.cdist(curr_embeds, allowed_embeddings)
    #dist[:, forbidden_toks] = float("inf")
    reg_loss = torch.min(dist, dim=1)[0].mean()

    return reg_loss