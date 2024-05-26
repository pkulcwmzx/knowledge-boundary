import torch
def detail_generate(model, tokenizer, embedding_table, input_embedding, attention_mask, device, generate_length):
    # generate the output of the model, return the generated tokens and logits and the hiddenstate of the last layer
    # input_embedding [bsz, seq_len, embed_dim]

    logits = []
    out_toks = []
    hidden_states = None

    for i in range(generate_length):
        out = model(inputs_embeds=input_embedding, attention_mask=attention_mask, output_hidden_states=True)
        if out.logits.shape[0] != input_embedding.shape[0]:
            logit = out.logits.transpose(0, 1)[:, -1, :]
        else:
            logit = out.logits[:, -1, :]
        logits.append(logit)
        out_tok = logit.argmax(dim=-1)
        out_toks.append(out_tok)
        hidden_states = out.hidden_states[-1]

        input_embedding = torch.cat([input_embedding, embedding_table[out_tok].unsqueeze(1)], dim=1)
        input_embedding = input_embedding.to(device)
        attention_mask = torch.cat([attention_mask, torch.ones([attention_mask.shape[0], 1]).long().to(device)], dim=1)
        attention_mask = attention_mask.to(device)

    logits = torch.stack(logits, dim=1)
    out_toks = torch.stack(out_toks, dim=1)
    return out_toks, logits, hidden_states



def generate_cand_tok(model, tokenizer, input_ids, tok_id, cand_num):
    """
    :param model: target model
    :param tokenizer: target model tokenizer
    :param input_ids: target input [batch_size, seq_len] (tensor)
    :param tok_id: the target token idx of the sequence
    :return: candidate token indices [batch_size, cand_num]
    """
    out = model(input_ids)
    logits = out.logits
    cand_logits = logits[:, :, tok_id - 1]
    _, indices = torch.topk(cand_logits, cand_num, dim=1)
    return indices