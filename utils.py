from datetime import datetime
import torch
import random
from torch import nn
import torch.nn.functional as F
from utils_loss import log_prob_loss, log_perplexity_loss
import copy
from sentence_transformers.util import (semantic_search,
                                        normalize_embeddings)


def batch_questions(questions, model, tokenizer):
    """

    :param questions: input questions for batch
    :param model: target model
    :param tokenizer: target tokenizer
    :return: batched questions in tensor
    """
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = '[PAD]'
    # question_num = len(questions)
    # (tokenizer.pad_token_id)
    padded_model_inputs = tokenizer(questions, padding="longest", truncation=True, return_tensors="pt")
    return padded_model_inputs

def find_topk(full_embeddings, model, tokenizer, labels):
    out = model(inputs_embeds=full_embeddings, labels=labels)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    logits = out.logits
    if torch.isnan(logits).any():
        assert False
    loss = log_prob_loss(out.logits, labels, loss_fn)

def update_tok(embedding_table, model, tokenizer, full_embeddings, tok_id, labels, lamb_perp=0):
    """

    :param model: target model
    :param tokenizer: target model tokenizer
    :param full_embeddings: target input [batch_size, seq_len, embed_dim] (tensor)
    :param tok_id: the target token idx of the sequence
    :return: scores [batch_size, vocab_size]
    """
    out = model(input_embeds=full_embeddings, labels=labels)
    logits = out.logits
    loss_fct = nn.CrossEntropyLoss(reduction='mean')
    loss = log_prob_loss(logits, labels, loss_fct)

    loss.backward(retain_graph=True)
    # 这里怎么计算还需要慎重 [batch_size, vocab_dim]
    backward_scores = -torch.matmul(full_embeddings.grad[:, tok_id, :], embedding_table.T)

    return backward_scores

def batch_index(target_tensor, index_tensor):
    # given a tensor target_tensor[a, b, c], we want to index the target tensor with index_tensor [a, b, d],
    # the returned tensor t[a, b, d] should satisfy t[i, j, k] = target_tensor[i, j, index_tensor[i, j, k]]
    if len(index_tensor.shape) == 3:
        shape_1, shape_2, shape_3 = index_tensor.shape
        range_tensor_1 = torch.Tensor([torch.full([shape_2, shape_3], i).tolist() for i in range(shape_1)]).long()
        range_tensor_2 = torch.Tensor([torch.full([shape_1, shape_3], i).tolist() for i in range(shape_2)]).transpose(0,
                                                                                                                      1).long()
        return target_tensor[range_tensor_1, range_tensor_2, index_tensor]

    elif len(index_tensor.shape) == 2:
        range_vector = torch.Tensor([[i] for i in range(index_tensor.shape[0])]).long()
        return target_tensor[range_vector, index_tensor]

def get_update_toks(args, full_embeddings, label_mask, out, forbidden_toks):
    # find top-k update position
    embedding_grad = torch.norm(full_embeddings.grad, dim=2, keepdim=True)
    # ensure ground-truth labels unchanged in update
    embedding_grad[label_mask] = 0
    # ensure bos token not changed
    embedding_grad[:, 0] = 0
    _, update_tok_id = torch.topk(embedding_grad, args.update_num, dim=1)
    update_tok_id = update_tok_id.squeeze(-1).long()
    # update_tok_id [sent_num, update_num]
    # calculate the candidate tokens
    range_vector = torch.Tensor([[i] for i in range(update_tok_id.shape[0])]).long()
    candidate_token_logits = out.logits[range_vector, update_tok_id]
    candidate_token_logits[:, :, forbidden_toks] = -float("inf")
    # candidate_token_logits [sent_num, update_num, vocab_size]
    _, candidate_tokens = torch.topk(candidate_token_logits, args.candidate_num, dim=-1)
    return update_tok_id, candidate_tokens, range_vector

def get_update_embedding(args, full_embeddings, embedding_table, update_tok_id, evaluate_indices):
    # each iteration evaluate (evaluate_num ^ update_num) cases
    test_cases = pow(args.evaluate_num, args.update_num)
    test_embeddings = full_embeddings.repeat(test_cases, 1, 1)
    evaluate_embeddings = embedding_table[evaluate_indices]
    batch_update_tok_id = update_tok_id.repeat(test_cases, 1)
    range_vector = torch.Tensor([[i] for i in range(batch_update_tok_id.shape[0])]).long()
    test_embeddings[range_vector, batch_update_tok_id] = evaluate_embeddings
    return test_embeddings, test_cases, range_vector, batch_update_tok_id


def score_to_indices(scores, candidate_tokens, num):
    # pick evaluate_num of tokens from the candidate tokens for evaluation and selection
    _, candidate_indices = torch.topk(scores, num, dim=-1)
    # change indices candidate_tokens [sent_num, update_num, candidate_num] each element is the token id
    # candidate_indices [sent_num, update_num, evaluate_num] each element is the indice of tokens in scores
    evaluate_indices = batch_index(candidate_tokens, candidate_indices)
    # evaluate_indices [sent_num, update_num, evaluate_num]
    evaluate_indices = evaluate_indices.tolist()
    return evaluate_indices


def get_success_prompt(update_tok_id, evaluate_indices, batch_update_tok_id, \
                       curr_tok_tensor, input_question_len, input_answer_len, \
                       questions, answers, answer_num, tokenizer, i):
    q_index = i % update_tok_id.shape[0]
    q_len = input_question_len[q_index]
    ans_len = input_answer_len[q_index]
    ans_question = curr_tok_tensor[q_index]
    ans_question[batch_update_tok_id[i]] = evaluate_indices[i]
    tgt_question = ans_question[:q_len]
    tgt_ans = ans_question[q_len: q_len + ans_len]
    tgt_question = tokenizer.decode(tgt_question)
    tgt_ans = tokenizer.decode(tgt_ans)

    ori_q_index = q_index // answer_num
    ori_a_index = q_index % answer_num
    ori_question = questions[ori_q_index]
    ori_ans = answers[ori_a_index]
    return tgt_question, tgt_ans, ori_question, ori_ans


def update_one_step(retain_indices, full_embeddings, test_embeddings, evaluate_indices, update_tok_id, curr_tok_tensor):
    for j in range(retain_indices.shape[0]):
        retain_indices[j] = j + retain_indices.shape[0] * retain_indices[j]
    full_embeddings = test_embeddings[retain_indices]
    # print(retain_indices, evaluate_indices)
    update_indices = evaluate_indices[retain_indices]

    for j in range(update_indices.shape[0]):
        # print(update_indices[j])
        curr_tok_tensor[j, update_tok_id[j]] = update_indices[j]
    return full_embeddings, curr_tok_tensor

def embedding_to_prompt(soft_embedding, embedding_table):
    # find the closest token to the embedding in the embedding table
    # soft_embedding [bsz, seq_len, embed_dim]
    # embedding_table [vocab_size, embed_dim]
    soft_embedding = soft_embedding.unsqueeze(2)
    myembedding_table = copy.deepcopy(embedding_table)
    myembedding_table = myembedding_table.unsqueeze(0)
    myembedding_table = myembedding_table / torch.norm(myembedding_table, dim=2, keepdim=True)
    similarity = torch.matmul(soft_embedding, myembedding_table.transpose(1, 2))
    similarity = similarity.squeeze(2)
    prompt = similarity.argmax(dim=2)
    # top 1 change to sample from top 5
    # prompt = torch.multinomial(similarity, 5)
    return prompt


def nn_project(curr_embeds, embedding_layer, forbidden_toks, ceil, device):
    bsz, seq_len, emb_dim = curr_embeds.shape

    # Using the sentence transformers semantic search which is
    # a dot product exact kNN search between a set of
    # query vectors and a corpus of vectors
    curr_embeds = curr_embeds.reshape((-1, emb_dim))
    curr_embeds = normalize_embeddings(curr_embeds)  # queries

    embedding_matrix = embedding_layer.weight
    embedding_matrix = normalize_embeddings(embedding_matrix)  # corpus
    #print(embedding_matrix.size())
    def l2_score(a, b):
        """
        Computes the l2 similarity (l2 norm of the element-wise difference) for each vector pair in a and b.

        :return: Matrix with res[i][j] = l2_norm(a[i] - b[j])
        """
        if len(a.shape) == 1:
            a = a.unsqueeze(0)
        if len(b.shape) == 1:
            b = b.unsqueeze(0)
        return 10-torch.cdist(a, b, p=2)

    hits = semantic_search(curr_embeds, embedding_matrix,
                           forbidden_toks=forbidden_toks,
                           query_chunk_size=curr_embeds.shape[0],
                           top_k=1,
                           score_function=l2_score)

    nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=device).reshape(bsz, seq_len)
    scores = torch.tensor([hit[0]["score"] for hit in hits], device=device).unsqueeze(1).repeat(1, emb_dim)

    projected_embeds = embedding_layer(nn_indices).reshape((-1, emb_dim))
    new_embedding = torch.where(scores > ceil, projected_embeds, curr_embeds).reshape((bsz, seq_len, emb_dim))

    return new_embedding, nn_indices